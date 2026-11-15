import os
import torch
from torch import nn
import torch.distributed as dist

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
     