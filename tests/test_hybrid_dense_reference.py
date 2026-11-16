"""Compare distributed Qwen3 updates against the unsharded model."""
import argparse
import math
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F

from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.parallel_mesh import ParallelMesh
from framework.model_composer import compose_model
from framework.hybrid_train import HybridTrainer, data_replica_count, replica_index, create_batch_group, broadcast_replica_batch
from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism
from framework.fsdp import ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedTensorParallelLinear


def dense_parameter_name(name, mesh):
    if name.startswith("layers."):
        match = re.match(r"layers\.(\d+)\.(.+)", name)
        assert match is not None
        # pp ranks own consecutive layers
        layer_start = mesh.rank("pp") * (4 // mesh.size("pp")) + min(mesh.rank("pp"), 4 % mesh.size("pp"))
        name = "transformerlayers." + str(layer_start + int(match.group(1))) + "." + match.group(2)
    match = re.search(r"\.moe\.experts\.(\d+)\.", name)
    if match is not None and mesh.size("ep") > 1:
        number = int(match.group(1)) + mesh.rank("ep") * (4 // mesh.size("ep"))
        name = name[:match.start(1)] + str(number) + name[match.end(1):]
    return name


def validate_parameter_weights(dense, distributed, mesh):
    dense_params = dict(dense.named_parameters())
    for name, module in distributed.named_modules():
        if isinstance(module, ShardedTensorParallelLinear):
            key = dense_parameter_name(name + ".weight", mesh)
            weight = dense_params[key].detach()
            tpdim = 0 if module.columnwise else 1
            weight = torch.tensor_split(weight, mesh.size("tp"), dim=tpdim)[mesh.rank("tp")]
            flat = weight.reshape(-1)
            size = math.ceil(flat.numel() / mesh.size("fsdp"))
            padded = F.pad(flat, (0, size * mesh.size("fsdp") - flat.numel()))
            expected = padded.reshape(-1)[mesh.rank("fsdp") * size:(mesh.rank("fsdp") + 1) * size]
            torch.testing.assert_close(module.weightshard.detach(), expected, rtol=1e-3, atol=5e-6, msg=key)
        elif isinstance(module, (ColumnwiseLinearParallelism, RowwiseLinearParallelism)):
            key = dense_parameter_name(name + ".weight", mesh)
            weight = dense_params[key].detach()
            dim = 0 if isinstance(module, ColumnwiseLinearParallelism) else 1
            expected = torch.tensor_split(weight, mesh.size("tp"), dim=dim)[mesh.rank("tp")]
            torch.testing.assert_close(module.weight.detach(), expected, rtol=1e-3, atol=5e-6, msg=key)
        elif isinstance(module, (ShardedLinear, ShardedEmbedding, ShardedRMSNorm)):
            key = dense_parameter_name(name + ".weight", mesh)
            weight = dense_params[key].detach()
            flat = weight.reshape(-1)
            size = math.ceil(flat.numel() / mesh.size("fsdp"))
            padded = F.pad(flat, (0, size * mesh.size("fsdp") - flat.numel()))
            expected = padded[mesh.rank("fsdp") * size:(mesh.rank("fsdp") + 1) * size]
            torch.testing.assert_close(module.weightshard.detach(), expected, rtol=1e-3, atol=5e-6, msg=key)
    tp_weight_names = {name + ".weight" for name, module in distributed.named_modules()
                       if isinstance(module, (ColumnwiseLinearParallelism, RowwiseLinearParallelism))}
    for name, param in distributed.named_parameters():
        if name.endswith(".weightshard") or name in tp_weight_names:
            continue
        if name in dense_params or name.startswith("layers."):
            expected = dense_params[dense_parameter_name(name, mesh)].detach()
            torch.testing.assert_close(param.detach(), expected, rtol=1e-3, atol=5e-6, msg=name)


def reference_batch(args, mesh, step, replica):
    generator = torch.Generator().manual_seed(42 + 100000 * replica + 1000 * step)
    return torch.randint(0, args.vocab_size, (2, args.max_seq_len + 1), generator=generator)


def main():
    parser = argparse.ArgumentParser()
    for axis in ParallelMesh.DIMENSIONS:
        parser.add_argument("--" + axis, default=1, type=int)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        mesh = ParallelMesh(**vars(opts))
        group, leader = create_batch_group(mesh)
        args = Qwen3ModelArgs(vocab_size=37, max_seq_len=8, dim=8, moe_inter_dim=12,
                             numlayers=4, numheads=4, numkvheads=2, head_dim=4,
                             experts=4, k=2)
        torch.manual_seed(42)
        dense = Qwen3Model(args)
        torch.manual_seed(42)
        model = compose_model(Qwen3Model(args), mesh, args.max_seq_len).model
        reference_optim = torch.optim.AdamW(dense.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = HybridTrainer(model, mesh, optimizer, max_norm=0.15, microbatches=2)
        validate_parameter_weights(dense, model, mesh)

        for step in range(2):
            tokens = reference_batch(args, mesh, step, replica_index(mesh)) if dist.get_rank() == leader else None
            batch = broadcast_replica_batch(tokens, mesh, group, leader, (2, 9), torch.device("cpu"))

            reference_optim.zero_grad(set_to_none=True)
            reference_losses = []
            for replica in range(data_replica_count(mesh)):
                dense_batch = reference_batch(args, mesh, step, replica)
                logits = dense(dense_batch[:, :-1])
                reference_losses.append(F.cross_entropy(logits.flatten(0, -2), dense_batch[:, 1:].flatten()))
            reference_loss = torch.stack(reference_losses).mean()
            reference_loss.backward()
            reference_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), 0.15)
            reference_optim.step()

            hybrid_loss, hybrid_norm = trainer.training_step(batch)
            torch.testing.assert_close(torch.tensor(hybrid_loss), reference_loss.detach(),
                                       rtol=2e-3, atol=5e-5, msg="hybrid loss")
            torch.testing.assert_close(torch.tensor(hybrid_norm), reference_norm,
                                       rtol=2e-3, atol=5e-5, msg="hybrid grad norm")
            validate_parameter_weights(dense, model, mesh)

        if mesh.global_rank == 0:
            print("PASS: hybrid matches dense Qwen3 loss, grad norm and AdamW parameters", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
