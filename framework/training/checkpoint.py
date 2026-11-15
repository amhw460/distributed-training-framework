from pathlib import Path
import torch.distributed as dist
from framework.fsdp import ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts
import torch



class Checkpoint():
  def __init__(self, trainer, folder, dataset):
    self.trainer = trainer
    self.folder = Path(folder)
    self.dataset = dataset


  def _require_unsharded_model(self):
    sharded_types = (ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts)

    if any(isinstance(module, sharded_types) for module in self.trainer.model.modules()):
        raise RuntimeError("FSDP checkpointing is not implemented yet.")

  def save(self, step):
    self._require_unsharded_model()
    if dist.is_initialized() and dist.get_rank() != 0:
        return
        
    state = (self.trainer.model.state_dict(), 
    self.trainer.optimizer.state_dict(), 
    self.trainer.scheduler.state_dict(), 
    self.trainer.state(), 
    torch.get_rng_state(),
    self.dataset.state())

    self.folder.mkdir(parents=True, exist_ok=True)
    path = self.folder / f"step-{step}.pt"
    torch.save(state, path)

  def load(self, path):
    self._require_unsharded_model()
    model, optimizer, scheduler, trainer, rng, dataset = torch.load(path, map_location='cpu', weights_only=True)

    self.trainer.model.load_state_dict(model)
    self.trainer.optimizer.load_state_dict(optimizer)
    self.trainer.scheduler.load_state_dict(scheduler)
    self.trainer.load_state(trainer)
    torch.set_rng_state(rng)
    self.dataset.load_state(dataset)
