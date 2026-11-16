import torch
import torch.nn.functional as F
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs

torch.manual_seed(0)
args = Qwen3ModelArgs(vocab_size=100, max_seq_len=16, dim=32, moe_inter_dim=64, numlayers=2, numheads=4, numkvheads=2, head_dim=8, experts=4, k=2)
model = Qwen3Model(args)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tokens = torch.randint(0, args.vocab_size, (2, 4))

logits = model(tokens[:, :-1])
assert logits.shape == (2, 3, args.vocab_size)

loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), tokens[:, 1:].reshape(-1))
optimizer.zero_grad()
loss.backward()

assert model.transformerlayers[0].attention.Wq.weight.grad is not None
assert model.transformerlayers[0].moe.router.router.weight.grad is not None
assert any(expert.up_proj.weight.grad is not None for expert in model.transformerlayers[0].moe.experts)

before = model.outputprojection.weight.detach().clone()
optimizer.step()
assert not torch.equal(before, model.outputprojection.weight)
print(f"Training step passed; loss = {loss.item():.4f}")
