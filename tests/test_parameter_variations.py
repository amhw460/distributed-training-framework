"""Matrix of small Qwen3 configurations against dense reference training.

Run:
  torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode tp
  torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode fsdp
  torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode ddp
No GPUs, tokenizer, or network downloads required.
"""
import argparse
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from framework.distributed import synchronize_model, synchronize_gradients
from framework.fsdp import shard_model
from framework.gradient_clipping import clip_hybrid_grad_norm_, clip_tp_grad_norm_
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.tensorparallel import paralellize_model, synchronize_tp_head_norm_gradients
from tests.test_full_fsdp import assert_sharded_weights_equal, assert_sharded_grads_equal
from tests.test_full_tp import compare as compare_tp


def cases(world):
    # Change depth, width, head grouping, expert count, top-k, and shard divisibility.
    return [
        Qwen3ModelArgs(vocab_size=43, max_seq_len=5,
            dim=8 + 4 * world, moe_inter_dim=3 * world + 2,
            numlayers=1, numheads=2 * world, numkvheads=world,
            head_dim=4, experts=3, k=1),
        Qwen3ModelArgs(vocab_size=53, max_seq_len=7,
            dim=12 + 4 * world, moe_inter_dim=5 * world + 1,
            numlayers=2, numheads=4 * world, numkvheads=world,
            head_dim=2, experts=5, k=2),
    ]


def loss_for(model, tokens, vocab):
    logits = model(tokens[:, :-1])
    labels = tokens[:, 1:].reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab), labels)
    return logits, loss


def verify_tp(args, rank, world, case_num):
    torch.manual_seed(300 + case_num)
    full = Qwen3Model(args)
    parallel = copy.deepcopy(full)
    paralellize_model(parallel)
    check = compare_tp

    opt_full = torch.optim.AdamW(full.parameters(), lr=0.003)
    opt_tp = torch.optim.AdamW(parallel.parameters(), lr=0.003)
    for step in range(2):
        opt_full.zero_grad()
        opt_tp.zero_grad()
        for micro in range(2):
            gen = torch.Generator().manual_seed(9500 + 20 * case_num + step * 2 + micro)
            tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len + 1), generator=gen)
            out_full, loss_full = loss_for(full, tokens, args.vocab_size)
            out_tp, loss_tp = loss_for(parallel, tokens, args.vocab_size)
            torch.testing.assert_close(out_tp, out_full, rtol=5e-4, atol=3e-6)
            (loss_full / 2).backward()
            (loss_tp / 2).backward()
        synchronize_tp_head_norm_gradients(parallel)
        check(full, parallel, rank, world)
        norm_full = torch.nn.utils.clip_grad_norm_(full.parameters(), 0.12)
        norm_tp = clip_tp_grad_norm_(parallel, 0.12)
        torch.testing.assert_close(torch.tensor(norm_tp), norm_full, rtol=5e-4, atol=3e-6)
        check(full, parallel, rank, world)
        opt_full.step()
        opt_tp.step()
        check(full, parallel, rank, world)


def verify_fsdp(args, rank, world, case_num):
    torch.manual_seed(1000 + case_num + rank)
    full = Qwen3Model(args)
    synchronize_model(full)
    sharded = copy.deepcopy(full)
    shard_model(sharded)
    assert_sharded_weights_equal(full, sharded)

    opt_full = torch.optim.AdamW(full.parameters(), lr=0.003)
    opt_fsdp = torch.optim.AdamW(sharded.parameters(), lr=0.003)
    for step in range(2):
        opt_full.zero_grad()
        opt_fsdp.zero_grad()
        for micro in range(2):
            gen = torch.Generator().manual_seed(9200 + 1000 * rank + 20 * case_num + step * 2 + micro)
            tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len + 1), generator=gen)
            out_full, loss_full = loss_for(full, tokens, args.vocab_size)
            out_fsdp, loss_fsdp = loss_for(sharded, tokens, args.vocab_size)
            torch.testing.assert_close(out_fsdp, out_full, rtol=5e-4, atol=3e-6)
            (loss_full / 2).backward()
            (loss_fsdp / 2).backward()
        synchronize_gradients(full)
        synchronize_gradients(sharded)
        assert_sharded_grads_equal(full, sharded, rank, world)
        norm_full = clip_hybrid_grad_norm_(full, 0.12)
        norm_fsdp = clip_hybrid_grad_norm_(sharded, 0.12)
        torch.testing.assert_close(torch.tensor(norm_fsdp), torch.tensor(norm_full), rtol=5e-4, atol=3e-6)
        opt_full.step()
        opt_fsdp.step()
        assert_sharded_weights_equal(full, sharded)


def verify_ddp(args, rank, world, case_num):
    torch.manual_seed(4000 + rank + case_num)
    distributed = Qwen3Model(args)
    synchronize_model(distributed)
    full = copy.deepcopy(distributed)
    opt_full = torch.optim.AdamW(full.parameters(), lr=0.003)
    opt_dist = torch.optim.AdamW(distributed.parameters(), lr=0.003)
    for step in range(2):
        opt_full.zero_grad()
        opt_dist.zero_grad()
        gen = torch.Generator().manual_seed(7700 + 1000 * rank + 20 * case_num + step)
        local_tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len + 1), generator=gen)
        received = [torch.empty_like(local_tokens) for _ in range(world)]
        dist.all_gather(received, local_tokens)
        all_tokens = torch.cat(received)
        _, loss_full = loss_for(full, all_tokens, args.vocab_size)
        _, loss_local = loss_for(distributed, local_tokens, args.vocab_size)
        loss_full.backward()
        loss_local.backward()
        synchronize_gradients(distributed)
        for name, ref in full.named_parameters():
            actual = dict(distributed.named_parameters())[name]
            assert (ref.grad is None) == (actual.grad is None), f"grad presence {name}"
            if ref.grad is not None:
                torch.testing.assert_close(actual.grad, ref.grad, rtol=5e-4, atol=3e-6, msg=name)
        torch.nn.utils.clip_grad_norm_(full.parameters(), 0.12)
        torch.nn.utils.clip_grad_norm_(distributed.parameters(), 0.12)
        opt_full.step()
        opt_dist.step()
        for name, ref in full.named_parameters():
            torch.testing.assert_close(dict(distributed.named_parameters())[name], ref, rtol=5e-4, atol=3e-6, msg=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tp", "fsdp", "ddp"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        for case_num, config in enumerate(cases(world)):
            if args.mode == "tp":
                verify_tp(config, rank, world, case_num)
            elif args.mode == "fsdp":
                verify_fsdp(config, rank, world, case_num)
            else:
                verify_ddp(config, rank, world, case_num)
            if rank == 0:
                print(f"PASS: {args.mode}, world={world}, config={case_num+1}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
