import torch
from torch import nn

class Expert(nn.Module):
  def __init__(self, model_args):
    super().__init__()
    self.model_args = model_args

    self.up_proj = nn.Linear(self.model_args.dim, self.model_args.moe_inter_dim, bias = False)
    self.down_proj = nn.Linear(self.model_args.moe_inter_dim, self.model_args.dim, bias = False)

    self.gate = nn.Linear(self.model_args.dim, self.model_args.moe_inter_dim, bias = False)


  def forward(self, token):
    # swiglu is used here as is in standard qwen
    # basically apply your standard up projection, apply the gate projection (and silu that), multiply, then down project

    up = self.up_proj(token)
    gate = self.gate(token)

    gate = nn.silu()(gate)

    combined = up * gate

    return self.down_proj(combined)

