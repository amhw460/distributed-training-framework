from dataclasses import dataclass

@dataclass 
class Qwen3ModelArgs:
  # temporarily, this is using a sized down version of Qwen-3-30b. Afterwards, I'll scale this back up to full model size. 

  max_seq_len: int = 4096
  vocab_size: int = 151936 

  dim: int = 1536
  moe_inter_dim:int = 1024

  numlayers: int = 27 # my favorite number lol
  numheads: int = 24
  numkvheads: int = 4
  head_dim: int = 128

  experts: int = 128
  k: int = 4

  norm_eps: float = 1e-6
  rope_theta: float = 1000000.0

  def _num_params_flops(self, model, sequence_len):
    expert_params = 0
    total_params = 0

    for name, parameter in model.named_parameters():
      total_params += parameter.numel()

      if 'moe.experts' in name:
        expert_params += parameter.numel()

    dense_params = total_params - expert_params
    active_params = dense_params + expert_params * self.k // self.experts

    flops_per_token = (6 * active_params) + (12 * self.numlayers * sequence_len 
                                             * self.numheads * self.head_dim)

    return total_params, active_params, flops_per_token
        
