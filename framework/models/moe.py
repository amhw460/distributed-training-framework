import torch
from torch import nn

from framework.fsdp import ShardedExperts


class Expert(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args

        self.up_proj = nn.Linear(
            self.model_args.dim, self.model_args.moe_inter_dim, bias=False
        )
        self.down_proj = nn.Linear(
            self.model_args.moe_inter_dim, self.model_args.dim, bias=False
        )
        self.gate = nn.Linear(
            self.model_args.dim, self.model_args.moe_inter_dim, bias=False
        )

    def forward(self, token):
        # swiglu: gate the up projection, then project back down
        up = self.up_proj(token)
        gate = self.gate(token)
        gate = nn.SiLU()(gate)
        combined = up * gate
        return self.down_proj(combined)


class Router(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.router = nn.Linear(
            self.model_args.dim, self.model_args.experts, bias=False
        )
        self.k = self.model_args.k

    def forward(self, token):
        expertlogits = self.router(token)
        expertprobs = torch.softmax(expertlogits.float(), dim=-1)

        # keep the top k experts and normalize their routing weights
        topkweights, topkindices = torch.topk(expertprobs, k=self.k, dim=-1)
        topkweights = topkweights / topkweights.sum(dim=-1, keepdim=True)

        return topkweights, topkindices, expertlogits


class MoE(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.router = Router(model_args)
        self.experts = nn.ModuleList([
            Expert(model_args) for _ in range(model_args.experts)
        ])

    def forward(self, tokens):
        batch = tokens.shape[0]
        seqlen = tokens.shape[1]
        tokenstates = tokens.reshape(
            tokens.shape[0] * tokens.shape[1], tokens.shape[2]
        )

        topkweights, topkindices, expertlogits = self.router(tokenstates)

        if isinstance(self.experts, ShardedExperts):
            output = self.experts(tokenstates, topkindices, topkweights)
            return output.reshape(tokens.shape)

        output = torch.zeros(
            batch * seqlen, self.model_args.dim,
            device=tokens.device, dtype=tokens.dtype,
        )

        # find the tokens routed to each expert and combine their weighted outputs
        for expert in range(self.model_args.experts):
            tokenindices, whichoftopk = torch.where(topkindices == expert)

            if tokenindices.numel() == 0 and not getattr(self, "fsdp_mode", False):
                continue

            toks = tokenstates[tokenindices]
            weights = topkweights[tokenindices, whichoftopk]
            weightedoutput = weights.unsqueeze(-1) * self.experts[expert](toks)
            output = output.index_add(
                0, tokenindices, weightedoutput.to(output.dtype)
            )

        output = output.reshape(tokens.shape)
        return output
