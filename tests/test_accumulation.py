
import copy

import torch
import torch.nn.functional as F

from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.training.trainer import Trainer
from framework.training.config import TrainingConfig

torch.manual_seed(42)

args = Qwen3ModelArgs(vocab_size=100, max_seq_len=8, dim=32, moe_inter_dim=64, numlayers=2, numheads=4, numkvheads=2, head_dim=8, experts=4, k=2)

model_large = Qwen3Model(args)
model_accum = copy.deepcopy(model_large)

optimizer_large = torch.optim.AdamW(model_large.parameters(), lr=1e-3)
optimizer_accum = torch.optim.AdamW(model_accum.parameters(), lr=1e-3)

# 8 sequences, each containing 9 tokens (8 inputs + 8 labels)
tokens = torch.randint(0, args.vocab_size, (8, 9))

# TEST A: One large batch of 8 sequences
optimizer_large.zero_grad()

logits = model_large(tokens[:, :-1])
loss_large = F.cross_entropy(logits.reshape(-1, args.vocab_size), tokens[:, 1:].reshape(-1))

loss_large.backward()
torch.nn.utils.clip_grad_norm_(model_large.parameters(), 1.0)
optimizer_large.step()

# TEST B: Four microbatches of 2 sequences each
# Use the existing Trainer methods directly.
config = TrainingConfig(steps=2, localbatchsize=2, globalbatchsize=8, seqlen=8, learningrate=1e-3, maxnorm=1.0, seed=42, warmupsteps=0, decayratio=1.0, decaytype="linear", min_lr=0.1)
trainer = Trainer(model_accum, optimizer_accum, "cpu", config)
assert trainer.steps_gradient_accumulation == 4

microbatches = list(tokens.split(2, dim=0))
loss_accum, lr, gradnorm, tokensseen, elapsed, tps = trainer.training_step(iter(microbatches))

# Compare all parameters
max_diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(model_large.parameters(), model_accum.parameters()))

print(f"Large batch loss:       {loss_large.item():.8f}")
print(f"Accumulated batch loss: {loss_accum:.8f}")
print(f"Maximum parameter diff: {max_diff:.10f}")

assert abs(loss_large.item() - loss_accum) < 1e-5
assert max_diff < 1e-5
assert trainer.trainingstep == 1
assert tokensseen == 64
assert lr == 1e-3
assert gradnorm > 0
assert elapsed > 0
assert tps == tokensseen / elapsed
assert trainer.scheduler.last_epoch == 1

print("PASS: Gradient accumulation matches large-batch training")

# train() should pass one shared iterator through each optimizer update.
trainer.train(iter(microbatches), 1)
assert trainer.trainingstep == 2
assert trainer.tokensseen == 128
assert trainer.scheduler.last_epoch == 2
print("PASS: Training loop consumes four microbatches per update")
