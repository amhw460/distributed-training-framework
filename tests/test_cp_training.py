import copy

import torch
import torch.distributed as dist

from framework.contextparallel import parallelize_cp
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.training.config import TrainingConfig
from framework.training.trainer import Trainer


def run():
  torch.set_num_threads(1)
  dist.init_process_group("gloo")
  try:
    rank, total_nodes = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)

    args = Qwen3ModelArgs(vocab_size=37, max_seq_len=4 * total_nodes,
                          dim=8, moe_inter_dim=12, numlayers=2,
                          numheads=4, numkvheads=2, head_dim=4, experts=4, k=2)
    config = TrainingConfig(steps=2, localbatchsize=2, globalbatchsize=4,
                            seqlen=args.max_seq_len, learningrate=0.003,
                            maxnorm=0.1, seed=42, warmupsteps=1,
                            decayratio=0.5, decaytype="cosine", min_lr=0.1)

    dense = Qwen3Model(args)
    cp = parallelize_cp(copy.deepcopy(dense))

    denseoptimizer = torch.optim.AdamW(dense.parameters(), lr=config.lr)
    cpoptimizer = torch.optim.AdamW(cp.parameters(), lr=config.lr)

    dense_trainer = Trainer(dense, denseoptimizer, torch.device("cpu"), config)
    cp_trainer = Trainer(cp, cpoptimizer, torch.device("cpu"), config, cp_mode=True)

    dense_trainer.steps_gradient_accumulation = 2
    assert cp_trainer.steps_gradient_accumulation == 2

    batches = [torch.randint(0, args.vocab_size, (2, args.max_seq_len + 1),
                             generator=torch.Generator().manual_seed(100 + i))
               for i in range(4)]

    for step in range(2):
      reference = dense_trainer.training_step(iter(batches[2 * step:2 * step + 2]))
      actual = cp_trainer.training_step(iter(batches[2 * step:2 * step + 2]))

      torch.testing.assert_close(torch.tensor(actual[0]), torch.tensor(reference[0]),
                                 rtol=5e-4, atol=3e-6, msg="loss")

      for name, parameter in cp.named_parameters():
        expected = dict(dense.named_parameters())[name]
        assert (parameter.grad is None) == (expected.grad is None), name
        if parameter.grad is not None:
          torch.testing.assert_close(parameter.grad, expected.grad,
                                     rtol=1e-3, atol=3e-6, msg="gradient " + name)
        torch.testing.assert_close(parameter, expected,
                                   rtol=5e-4, atol=3e-6, msg="weight " + name)

    if rank == 0:
      print(f"PASS: CP world={total_nodes}, training loss, gradients, clipping and AdamW", flush=True)
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  run()
