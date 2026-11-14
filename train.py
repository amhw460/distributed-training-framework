import torch
from torch import nn

class Trainer():
  def __init__(self, model, optimizer, device, config):

    self.model = model.to(device)
    self.optimizer = optimizer
    self.device = device
    self.config = config
    if self.config.local_batch_size <= 0 or self.config.global_batch_size <= 0 or self.config.global_batch_size % self.config.local_batch_size != 0:
      raise ValueError("global batch size must be a positive multiple of local batch size")
    self.steps_gradient_accumulation = self.config.global_batch_size // self.config.local_batch_size

    self.trainingstep = 0

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

    self.optimizer.zero_grad() 

    totalloss:float = 0

    for step in range(self.steps_gradient_accumulation):
      currentbatch = next(data_iterator)
      loss = self.forwardbackwardpass(currentbatch)
      
      totalloss += loss

    avgloss = totalloss / self.steps_gradient_accumulation 

    # cliip gradients if they exceed max
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.maxnorm)

    self.optimizer.step()
    self.trainingstep += 1

    return avgloss

  def train(self, batches, steps):
    iterator = iter(batches)

    for step in range(steps):
      loss = self.training_step(iterator)
      print(f"loss = {loss}")

