import torch
from torch import nn
from model_args import Qwen3ModelArgs

class Qwen3Model(nn.Module):
    def _rotation(self):
            indices = torch.arange(self.model_args.head_dim // 2)
    
            rotation_freq = 1 / torch.pow(self.model_args.rope_theta, 
                                                    2 * indices / self.model_args.head_dim)
    
            positions = torch.arange(self.model_args.max_seq_len)
    
            rotation_angle = torch.outer(positions, rotation_freq)
    
            cosine = torch.cos(rotation_angle)
            sine = torch.sin(rotation_angle)   
    
            return (cosine, sine)
    
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.embeddings = nn.Embedding(num_embeddings = self.model_args.vocab_size, embedding_dim = self.model_args.dim)

        cosine, sine = self._rotation()

        # we can precompute all the cosine, sine rotations for all possible rope embeddings once and just access it later on
        # however we want to use a buffer to make sure it moves devices 

        self.register_buffer("rope_cosine_values", cosine, persistent=False)
        self.register_buffer("rope_sine_values", sine, persistent=False)
          
    def _RoPE(self, vector): 
        split = torch.split(vector, self.model_args.head_dim // 2, dim = -1)
        rope_pairs = torch.stack((split[0], split[1]), dim = -1)

        cosine, sine = self.rope_cosine_values, self.rope_sine_values
        cosine, sine = cosine[:vector.shape[1]], sine[:vector.shape[1]]
        cosine, sine = cosine[None, :, None, :], sine[None, :, None, :]

        newx = rope_pairs[..., 0] * cosine - rope_pairs[..., 1] * sine
        newy = rope_pairs[..., 0] * sine + rope_pairs[..., 1] * cosine

        return torch.cat((newx, newy), dim=-1)

