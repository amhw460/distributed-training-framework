"""Compare two-rank tensor-parallel Qwen3 against the full dense model.

Run:
 torchrun --standalone --nproc-per-node=2 -m tests.test_full_tp
"""
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism, paralellize_model
from framework.gradient_clipping import clip_tp_grad_norm_


def compare(dense, tp, rank, world):
    original = dict(dense.named_parameters())
    replicas = {}
    for name, module in tp.named_modules():
        if isinstance(module, (ColumnwiseLinearParallelism, RowwiseLinearParallelism)):
            full_name = name + ".weight"
            full = original[full_name]
            dim = 0 if isinstance(module, ColumnwiseLinearParallelism) else 1
            expected = torch.tensor_split(full.detach(), world, dim=dim)[rank]
            torch.testing.assert_close(module.weight.detach(), expected, rtol=1e-4, atol=2e-6, msg=f"weight: {name}")
            if full.grad is not None:
                actual_grad = module.weight.grad
                assert actual_grad is not None, f"Missing TP gradient {name}"
                torch.testing.assert_close(actual_grad, torch.tensor_split(full.grad, world, dim=dim)[rank],
                    rtol=3e-4, atol=2e-6, msg=f"gradient: {name}")
        elif name + ".weight" in original and hasattr(module, "weight") and module.weight is not None:
            replicas[name + ".weight"] = module.weight
    for name, local_param in replicas.items():
        ref = original[name]
        torch.testing.assert_close(local_param.detach(), ref.detach(), rtol=1e-4, atol=2e-6, msg=f"replicated weight {name}")
        if ref.grad is not None:
            assert local_param.grad is not None, f"Missing replicated gradient {name}"
            torch.testing.assert_close(local_param.grad, ref.grad, rtol=3e-4, atol=2e-6, msg=f"replicated gradient {name}")


def run():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        assert world == 2
        args = Qwen3ModelArgs(vocab_size=37, max_seq_len=6, dim=8, moe_inter_dim=12,
            numlayers=2, numheads=4, numkvheads=2, head_dim=4, experts=4, k=2)
        torch.manual_seed(114)
        dense = Qwen3Model(args)
        tp = copy.deepcopy(dense)
        paralellize_model(tp)
        optim_dense = torch.optim.AdamW(dense.parameters(), lr=0.003, weight_decay=0.01)
        optim_tp = torch.optim.AdamW(tp.parameters(), lr=0.003, weight_decay=0.01)
        for step in range(2):
            optim_dense.zero_grad()
            optim_tp.zero_grad()
            for micro in range(2):
                gen = torch.Generator().manual_seed(3300 + step*10 + micro)
                tokens = torch.randint(0, args.vocab_size, (2,6), generator=gen)
                x,y=tokens[:,:-1],tokens[:,1:]
                dense_logits=dense(x)
                tp_logits=tp(x)
                torch.testing.assert_close(tp_logits, dense_logits, rtol=3e-4, atol=2e-6,
                    msg=f"Forward outputs mismatch: step={step} micro={micro}")
                (F.cross_entropy(dense_logits.reshape(-1,args.vocab_size),y.reshape(-1))/2).backward()
                (F.cross_entropy(tp_logits.reshape(-1,args.vocab_size),y.reshape(-1))/2).backward()
            compare(dense,tp,rank,world)
            ref_norm = torch.nn.utils.clip_grad_norm_(dense.parameters(), max_norm=0.1)
            tp_norm = clip_tp_grad_norm_(tp, max_norm=0.1)
            torch.testing.assert_close(torch.tensor(tp_norm), ref_norm, rtol=3e-4, atol=2e-6,
                msg=f"Global gradient norm mismatch step {step}")
            compare(dense,tp,rank,world)
            optim_dense.step()
            optim_tp.step()
            compare(dense,tp,rank,world)
        if rank == 0:
            print("PASS: full TP matches dense Qwen3 in logits, gradients, clipping, and two AdamW steps",flush=True)
    finally:
        dist.destroy_process_group()


if __name__=="__main__":
    run()
