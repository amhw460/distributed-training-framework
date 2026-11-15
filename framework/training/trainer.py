import torch
from torch import nn
import time
from framework.training.lr_scheduler import build_lr_scheduler
from framework.distributed import synchronize_gradients
from framework.gradient_clipping import clip_hybrid_grad_norm_
import torch.distributed as dist
from framework.gradient_clipping import clip_tp_grad_norm_

class Trainer():
  def __init__(self, model, optimizer, device, config, tp_mode=False):

    self.model = model.to(device)
    self.optimizer = optimizer
    self.device = device
    self.config = config
    
    self.tp_mode = tp_mode #tensor parallel

    self.tokensseen = 0 # tokens processed
    self.trainingstep = 0

    self.scheduler = build_lr_scheduler(self.optimizer, self.config)

    self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    batch_multiplier = 1 if self.tp_mode else self.world_size

    effective_local_batch = self.config.local_batch_size * batch_multiplier

    if (self.config.local_batch_size <= 0 or self.config.global_batch_size <= 0 or self.config.global_batch_size % effective_local_batch != 0):
        raise ValueError("Invalid global batch size")

    self.steps_gradient_accumulation = self.config.global_batch_size // effective_local_batch

      

  def state(self): # we want to be able to resume if it gets blocked at some point
    return self.trainingstep, self.tokensseen

  def load_state(self, state):
    steps, tokens = state

    self.trainingstep, self.tokensseen = steps, tokens

  def forwardbackwardpass(self, tokens):
    tokens = tokens.to(self.device)

    inputs = tokens[:, :-1]
    labels = tokens[:, 1: ]

    logits = self.model(inputs)
    logits = logits.reshape(-1, logits.shape[-1])

    labels = labels.reshape(-1)

    loss = nn.CrossEntropyLoss()(logits, labels)

    loss = loss / self.steps_gradient_accumulation
    loss.backward()

    return loss.item() * self.steps_gradient_accumulation

  def training_step(self, data_iterator):

    start = time.perf_counter()
    lr = self.optimizer.param_groups[0]["lr"]
    tokens_this_step = 0

    self.optimizer.zero_grad() 

    totalloss:float = 0

    for step in range(self.steps_gradient_accumulation):
      currentbatch = next(data_iterator)
      loss = self.forwardbackwardpass(currentbatch)
      
      totalloss += loss
      tokens_this_step += (currentbatch.shape[0] * (currentbatch.shape[1] - 1))

    if self.world_size > 1 and dist.is_initialized() and not self.tpmode:
      synchronize_gradients(self.model) # sync after training, if its tensor parallel each rank has diff part of weight matrix

    avgloss = totalloss / self.steps_gradient_accumulation 

    # cliip gradients if they exceed max
    if self.tp_mode:
      gradient_norm = clip_tp_grad_norm_(self.model, self.config.maxnorm)
    else:
      gradient_norm = clip_hybrid_grad_norm_(self.model, self.config.maxnorm)

    self.optimizer.step()
    self.scheduler.step()

    self.trainingstep += 1

    elapsed_time = time.perf_counter() - start
    self.tokensseen += tokens_this_step

    tps = tokens_this_step / elapsed_time

    return avgloss, lr, gradient_norm, self.tokensseen, elapsed_time, tps

  def train(self, batches, steps, checkpoint=None, save_every =None):

    if checkpoint is not None and (save_every is None or save_every <= 0):
      raise ValueError("save_every must be positive when checkpointing")

    iterator = iter(batches)

    for step in range(steps):
      loss, lr, gradnorm, totaltokens, time, tps = self.training_step(iterator)
      print(f"loss = {loss}, learningrate = {lr}, gradient norm = {gradnorm}, total tokens seen = {totaltokens}, time per step = {time}, tokens/sec = {tps}")

      if checkpoint is not None and self.trainingstep % save_every == 0:
        checkpoint.save(self.trainingstep)
