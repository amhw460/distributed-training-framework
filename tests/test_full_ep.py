import torch
import torch.distributed as dist

from framework.models.moe import MoE
from framework.models.model_args import Qwen3ModelArgs
from framework.expertparallel import ExpertParallel, synchronize_ep_gradients, clip_ep_grad_norm_


def run():
  torch.set_num_threads(1)
  dist.init_process_group("gloo")
  try:
    rank, total_nodes = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)

    args = Qwen3ModelArgs(dim=8, moe_inter_dim=16, experts=4 * total_nodes, k=2)
    dense = MoE(args)
    ep = ExpertParallel(MoE(args), rank, total_nodes)
    ep.router.load_state_dict(dense.router.state_dict())
    for i, expert in enumerate(ep.experts):
      expert.load_state_dict(dense.experts[ep.start + i].state_dict())

    denseoptimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    epoptimizer = torch.optim.SGD(ep.parameters(), lr=0.01)

    for step in range(2):
      generator = torch.Generator().manual_seed(100 + step * total_nodes + rank)
      tokens = torch.randn(2, 3, args.dim, generator=generator)
      alltokens = [torch.empty_like(tokens) for _ in range(total_nodes)]
      dist.all_gather(alltokens, tokens)

      denseoptimizer.zero_grad()
      epoptimizer.zero_grad()

      # Each rank processes different tokens, but dense trains on the entire batch.
      outputs = dense(torch.cat(alltokens, dim=0))
      local = ep(tokens)
      torch.testing.assert_close(local, outputs[rank * 2:(rank + 1) * 2],
                                 rtol=1e-4, atol=1e-6)

      (outputs.square().mean()).backward()
      (local.square().mean()).backward()
      synchronize_ep_gradients(ep)

      denseparams = dict(dense.named_parameters())
      for name, parameter in ep.named_parameters():
        original = ("experts." + str(ep.start + int(name.split(".")[1])) +
                    "." + ".".join(name.split(".")[2:])) if name.startswith("experts.") else name
        ref = denseparams[original]
        if parameter.grad is None and ref.grad is None: continue
        assert parameter.grad is not None and ref.grad is not None, original
        torch.testing.assert_close(parameter.grad, ref.grad, rtol=5e-4, atol=2e-6,
                                   msg="gradient " + original)

      refnorm = torch.nn.utils.clip_grad_norm_(dense.parameters(), 0.1)
      epnorm = clip_ep_grad_norm_(ep, 0.1)
      torch.testing.assert_close(torch.tensor(epnorm, dtype=refnorm.dtype), refnorm,
                                 rtol=5e-4, atol=2e-6)
      denseoptimizer.step()
      epoptimizer.step()

      for name, parameter in ep.named_parameters():
        original = ("experts." + str(ep.start + int(name.split(".")[1])) +
                    "." + ".".join(name.split(".")[2:])) if name.startswith("experts.") else name
        torch.testing.assert_close(parameter, denseparams[original], rtol=5e-4, atol=2e-6,
                                   msg="weight " + original)

    if rank == 0:
      print(f"PASS: EP world={total_nodes} matches dense forward, gradients, clipping and optimizer", flush=True)
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  run()
