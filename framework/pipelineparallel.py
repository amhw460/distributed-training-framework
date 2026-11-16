import os
import torch
from torch import nn
import torch.distributed as dist
from framework.gradient_clipping import clip_pp_grad_norm_

class PipelineStage(nn.Module):
  def __init__(self, model, rank, total_nodes):
    super().__init__()

    nlayers = len(model.transformerlayers)

    if not 0 <= rank < total_nodes: raise ValueError('invalid rank')
    if not 1 <= total_nodes <= nlayers: raise ValueError('total amount of nodes must be betwee 1 and n layers')

    self.rank = rank
    self.total_nodes = total_nodes

    # each rank does its assigned layers unless its 0 / last rank otherwise it does like the embeddings / output head 

    # split up layers
    q, r = divmod(nlayers, total_nodes)
    start = rank * q + min(rank, r)
    count = q + (rank < r)
    end = start + count

    self.layers = nn.ModuleList(model.transformerlayers[start:end])

    self.embeddings = model.embeddings if rank == 0 else None # only rank 0 has embeddings

    self.finalRMSnorm, self.outputprojection = None, None

    if rank == total_nodes - 1:
      self.finalRMSnorm, self.outputprojection = model.finalRMSnorm, model.outputprojection

    self.register_buffer("rope_cosine_values", model.rope_cosine_values.clone(), persistent=False)
    self.register_buffer("rope_sine_values", model.rope_sine_values.clone(), persistent=False)

  def forward(self, x):
    tensor = self.embeddings(x) if self.embeddings is not None else x

    for layer in self.layers:
      tensor = layer(tensor, self.rope_cosine_values, self.rope_sine_values)

    if self.outputprojection is not None:
      tensor = self.outputprojection(self.finalRMSnorm(tensor))

    return tensor

# Each call processes exactly one microbatch on this stage.
def forward_microbatch(stage, x, rank, total_nodes, batch_size, seq_len, dim, device, dtype=torch.float32, group=None):
    if rank > 0:
        x = torch.empty((batch_size, seq_len, dim), device=device, dtype=dtype)
        dist.recv(x, src=dist.get_global_rank(group, rank - 1) if group is not None else rank - 1, group=group)
        x.requires_grad_(True)

    output = stage(x)
    if rank < total_nodes - 1:
        dist.send(output.detach().contiguous(), dst=dist.get_global_rank(group, rank + 1) if group is not None else rank + 1, group=group)
    return x, output


def backward_microbatch(x, output, rank, total_nodes, loss=None, group=None):
    if rank == total_nodes - 1:
        loss.backward()
    else:
        grad_output = torch.empty_like(output)
        dist.recv(grad_output, src=dist.get_global_rank(group, rank + 1) if group is not None else rank + 1, group=group)
        output.backward(grad_output)

    if rank > 0:
        dist.send(x.grad.contiguous(), dst=dist.get_global_rank(group, rank - 1) if group is not None else rank - 1, group=group)


def gpipe_step(stage, tokens, optimizer, num_microbatches, rank, total_nodes, dim, device, max_norm=None, group=None):
    # Rank 0 supplies the batch; all ranks need microbatch sizes, last rank needs labels.
    if rank == 0:
        tokens = tokens.to(device=device, dtype=torch.long).contiguous()
        shape = torch.tensor(tokens.shape, dtype=torch.long, device=device)
    else:
        shape = torch.empty(2, dtype=torch.long, device=device)

    if total_nodes > 1: dist.broadcast(shape, src=dist.get_global_rank(group, 0) if group is not None else 0, group=group)
    batch_size, seq_len = shape.tolist()
    if not 1 <= num_microbatches <= batch_size: raise ValueError("invalid microbatch count")
    if seq_len < 2: raise ValueError("sequence length must be at least 2")

    if rank != 0: tokens = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    if total_nodes > 1: dist.broadcast(tokens, src=dist.get_global_rank(group, 0) if group is not None else 0, group=group)

    stage.train()
    optimizer.zero_grad(set_to_none=True)
    microbatches = torch.tensor_split(tokens, num_microbatches, dim=0)
    saved = []
    dtype = next(stage.parameters()).dtype
    total_tokens = batch_size * (seq_len - 1)

    # GPipe: all forwards, then all backwards in reverse order.
    for mb in microbatches:
        x, output = forward_microbatch(stage, mb[:, :-1] if rank == 0 else None,
                                       rank, total_nodes, mb.size(0), seq_len - 1, dim, device, dtype, group=group)
        loss = None
        if rank == total_nodes - 1:
            loss = torch.nn.functional.cross_entropy(output.reshape(-1, output.size(-1)),
                                                      mb[:, 1:].reshape(-1), reduction="sum") / total_tokens
        saved.append((x, output, loss))

    for x, output, loss in reversed(saved):
        backward_microbatch(x, output, rank, total_nodes, loss, group=group)

    if max_norm is not None:
        clip_pp_grad_norm_(stage, max_norm, group=group)
    optimizer.step()
    if rank == total_nodes - 1:
        return sum(loss.detach().item() for _, _, loss in saved)
    return None
