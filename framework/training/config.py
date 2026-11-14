from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class TrainingConfig():
  def __init__(self, steps, localbatchsize, globalbatchsize, seqlen, learningrate, maxnorm, seed, warmupsteps, decayratio, decaytype, min_lr):
    
    self.steps = steps
    self.local_batch_size = localbatchsize
    self.global_batch_size = globalbatchsize
    self.seqlen = seqlen
    self.lr = learningrate
    self.maxnorm = maxnorm
    self.seed = seed

    # stuff for learning rate scheduling

    self.warmup_steps = warmupsteps
    self.decay_ratio = decayratio
    self.decay_type = decaytype
    self.min_lr = min_lr
