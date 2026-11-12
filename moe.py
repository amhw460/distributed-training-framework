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

    gate = nn.SiLU()(gate)

    combined = up * gate

    return self.down_proj(combined)

class Router(nn.Module):
  def __init__(self, model_args):
    super().__init__()

    self.model_args = model_args
    self.router = nn.Linear(self.model_args.dim, self.model_args.experts, bias = False)
    self.k = self.model_args.k

  def forward(self, token):

    expertlogits = self.router(token)
    expertprobs = torch.softmax(expertlogits.float(), dim=-1)

    # get top k experts

    topkweights, topkindices = torch.topk(expertprobs, k = self.k, dim = -1)

    topkweights = topkweights / topkweights.sum(dim = -1, keepdim = True)

    return topkweights, topkindices, expertlogits
  


