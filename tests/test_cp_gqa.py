import copy

import torch
import torch.distributed as dist

from framework.contextparallel import parallelize_cp
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs


def run():
  torch.set_num_threads(1)
  dist.init_process_group("gloo")
  try:
    rank, total_nodes = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)

    args = Qwen3ModelArgs(vocab_size=37, max_seq_len=total_nodes * 4,
                          dim=8, moe_inter_dim=12, numlayers=2,
                          numheads=4, numkvheads=2, head_dim=4, experts=4, k=2)
    dense = Qwen3Model(args)
    cp = parallelize_cp(copy.deepcopy(dense))

    denseoptimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    cpoptimizer = torch.optim.SGD(cp.parameters(), lr=0.01)

    for step in range(2):
      tokens = torch.randint(0, args.vocab_size, (2, args.max_seq_len),
                             generator=torch.Generator().manual_seed(100 + step))
      localtokens = tokens.tensor_split(total_nodes, dim=1)[rank]

      denseoptimizer.zero_grad()
      cpoptimizer.zero_grad()

      reference = dense(tokens)
      actual = cp(localtokens)

      torch.testing.assert_close(actual, reference.tensor_split(total_nodes, dim=1)[rank],
                                 rtol=5e-4, atol=3e-6, msg="full model logits")

      # Each rank owns 1/N of the loss; summing weight grads gives dense gradient.
      reference.square().mean().backward()
      (actual.square().mean() / total_nodes).backward()

      for name, param in cp.named_parameters():
        if param.grad is None: param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        expected = dict(dense.named_parameters())[name].grad
        torch.testing.assert_close(param.grad, expected, rtol=1e-3, atol=3e-6,
                                   msg="gradient " + name)

      denseoptimizer.step()
      cpoptimizer.step()
      for name, param in cp.named_parameters():
        torch.testing.assert_close(param, dict(dense.named_parameters())[name],
                                   rtol=5e-4, atol=3e-6, msg="weight " + name)

    if rank == 0:
      print(f"PASS: CP world={total_nodes} Qwen3 forward, backward, and optimizer", flush=True)
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  run()
