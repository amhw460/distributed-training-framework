"""Make sure each parallelism strategy uses the right communication group."""
import argparse
import copy

import torch
import torch.distributed as dist
from torch import nn

from framework.parallel_mesh import ParallelMesh
from framework.distributed import synchronize_gradients, synchronize_model
from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism
from framework.fsdp import ShardedLinear
from framework.contextparallel import ring_attention
from framework.expertparallel import parallelize_ep, synchronize_ep_gradients
from framework.pipelineparallel import forward_microbatch, backward_microbatch
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs


def test_dp(mesh):
    rank = dist.get_rank()
    source = mesh.ranks("dp")[0]
    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(-1)
    if rank == source: model.weight.data.fill_(float(source + 1))
    synchronize_model(model, group=mesh.group("dp"))
    assert torch.all(model.weight == float(source + 1))
    model.weight.grad = torch.full_like(model.weight, float(rank + 1))
    synchronize_gradients(model, group=mesh.group("dp"))
    expected = sum(r + 1 for r in mesh.ranks("dp")) / mesh.size("dp")
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, expected))


def test_tp(mesh):
    torch.manual_seed(30 + mesh.rank("dp"))
    group = mesh.group("tp")
    linear = nn.Linear(8, 8, bias=False)
    col = ColumnwiseLinearParallelism(linear, group=group)
    row = RowwiseLinearParallelism(linear, group=group)
    x = torch.randn(2, 8, requires_grad=True)
    x_col = x.detach().clone().requires_grad_()
    x_row = torch.randn(2, 4, requires_grad=True)

    out_col = col(x_col)
    expected_col = x @ linear.weight.T
    start = mesh.rank("tp") * 4
    torch.testing.assert_close(out_col, expected_col[:, start:start + 4])
    out_col.sum().backward()
    # input gradients should be summed across tp ranks
    torch.testing.assert_close(x_col.grad, linear.weight.sum(dim=0).expand_as(x_col))

    # rowwise tp sums the partial outputs
    rank_input = x[:, start:start + 4].detach().clone().requires_grad_()
    out_row = row(rank_input)
    torch.testing.assert_close(out_row, expected_col)
    out_row.square().sum().backward()
    torch.testing.assert_close(row.weight.grad,
                                                          (2 * expected_col).T @ x[:, start:start + 4], rtol=1e-5, atol=1e-6)


def test_fsdp(mesh):
    torch.manual_seed(50 + mesh.rank("dp"))
    full = nn.Linear(8, 8, bias=False)
    sharded = ShardedLinear(copy.deepcopy(full), group=mesh.group("fsdp"))
    x = torch.randn(3, 8)
    reference = full(x)
    output = sharded(x)
    torch.testing.assert_close(output, reference)
    reference.square().mean().backward()
    output.square().mean().backward()
    start = mesh.rank("fsdp") * 32
    torch.testing.assert_close(sharded.weightshard.grad,
                                                          full.weight.grad.flatten()[start:start + 32],
                                                          rtol=1e-5, atol=1e-6)


def test_cp(mesh):
    group = mesh.group("cp")
    torch.manual_seed(70 + mesh.rank("dp"))
    fullq = torch.randn(1, 2, 8, 4, requires_grad=True)
    fullk = torch.randn(1, 2, 8, 4, requires_grad=True)
    fullv = torch.randn(1, 2, 8, 4, requires_grad=True)
    localrank = mesh.rank("cp")
    q = fullq.detach().tensor_split(2, dim=2)[localrank].contiguous().requires_grad_()
    k = fullk.detach().tensor_split(2, dim=2)[localrank].contiguous().requires_grad_()
    v = fullv.detach().tensor_split(2, dim=2)[localrank].contiguous().requires_grad_()
    out = ring_attention(q, k, v, group=group)
    reference = torch.nn.functional.scaled_dot_product_attention(fullq, fullk, fullv, is_causal=True)
    torch.testing.assert_close(out, reference.tensor_split(2, dim=2)[localrank], rtol=1e-5, atol=1e-6)
    reference.square().mean().backward()
    (out.square().mean() / 2).backward()
    torch.testing.assert_close(q.grad, fullq.grad.tensor_split(2, dim=2)[localrank], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(k.grad, fullk.grad.tensor_split(2, dim=2)[localrank], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(v.grad, fullv.grad.tensor_split(2, dim=2)[localrank], rtol=1e-4, atol=1e-6)


def test_ep(mesh):
    torch.manual_seed(90 + mesh.rank("dp"))
    args = Qwen3ModelArgs(vocab_size=23, max_seq_len=4, dim=8, moe_inter_dim=12,
                                                  numlayers=1, numheads=4, numkvheads=2, head_dim=4, experts=4, k=2)
    dense = Qwen3Model(args)
    ep = parallelize_ep(copy.deepcopy(dense), group=mesh.group("ep"))
    generator = torch.Generator().manual_seed(300 + mesh.rank("dp") * 10 + mesh.rank("ep"))
    tokens = torch.randint(0, args.vocab_size, (2, 4), generator=generator)
    output = ep(tokens)
    ref = dense(tokens)
    torch.testing.assert_close(output, ref, rtol=1e-4, atol=1e-6)
    ref.square().mean().backward()
    output.square().mean().backward()
    synchronize_ep_gradients(ep, group=mesh.group("ep"))
    # average the reference gradients across ep ranks
    for name, p in dense.named_parameters():
        if p.grad is None: p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, group=mesh.group("ep"))
        p.grad.div_(mesh.size("ep"))
    for name, p in ep.named_parameters():
        if ".experts." in name:
            index = int(name.split(".experts.")[1].split(".")[0])
            dense_name = name.replace(f".experts.{index}.", f".experts.{index + mesh.rank('ep') * 2}.")
        else:
            dense_name = name
        torch.testing.assert_close(p.grad, dict(dense.named_parameters())[dense_name].grad, rtol=3e-4, atol=2e-6)


def test_pp(mesh):
    group = mesh.group("pp")
    localrank = mesh.rank("pp")
    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(float(mesh.rank("dp") + 1))
    # send activations forward and gradients backward
    if localrank == 0:
        x = torch.full((2, 3, 4), float(mesh.rank("dp") + 1))
        x.requires_grad_()
    else:
        x = None
    input_tensor, output = forward_microbatch(model, x, localrank, 2, 2, 3, 4,
                                                                                        torch.device("cpu"), group=group)
    if localrank == 1:
        loss = output.square().mean()
    else:
        loss = None
    backward_microbatch(input_tensor, output, localrank, 2, loss=loss, group=group)
    assert model.weight.grad is not None
    if localrank == 0: assert x.grad is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dp", "tp", "fsdp", "cp", "ep", "pp"), required=True)
    options = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        dimensions = {"dp": 2}
        if options.mode == "dp": dimensions["tp"] = 2
        else: dimensions[options.mode] = 2
        mesh = ParallelMesh(**dimensions)
        globals()["test_" + options.mode](mesh)
        if dist.get_rank() == 0: print("PASS: " + options.mode + " subgroup isolation", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
