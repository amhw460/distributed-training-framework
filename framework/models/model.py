from framework.models.moe import MoE
import torch
from torch import nn
from framework.models.model_args import Qwen3ModelArgs
from framework.contextparallel import ring_attention

def _RoPE(vector, cosine, sine, head_dim):
    split = torch.split(vector, head_dim // 2, dim = -1)
    rope_pairs = torch.stack((split[0], split[1]), dim = -1)
    
    cosine, sine = cosine[:vector.shape[1]], sine[:vector.shape[1]]
    cosine, sine = cosine[None, :, None, :], sine[None, :, None, :]
    
    newx = rope_pairs[..., 0] * cosine - rope_pairs[..., 1] * sine
    newy = rope_pairs[..., 0] * sine + rope_pairs[..., 1] * cosine
    
    return torch.cat((newx, newy), dim=-1)
     
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

        self.transformerlayers = nn.ModuleList([
             Transformer(self.model_args) for i in range(self.model_args.numlayers)
        ])

        self.finalRMSnorm = nn.RMSNorm(self.model_args.dim, eps = model_args.norm_eps)
        self.outputprojection = nn.Linear(self.model_args.dim, self.model_args.vocab_size, bias = False)

        self.apply(self._init_weights)

    def _init_weights(self, module):

        # using qwens specific weight initialization method to be more 'faithful'
      if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=self.model_args.initializer_range)

    def forward(self, tokenIDs):

         # forward pass: turn into embeddings, run thru transformer blocks, RMSnorm, then output project to the end vocab size

         embeddings = self.embeddings(tokenIDs)

         offset = getattr(self, "cp_rank", 0) * tokenIDs.shape[1]
         cosine = self.rope_cosine_values[offset:offset + tokenIDs.shape[1]]
         sine = self.rope_sine_values[offset:offset + tokenIDs.shape[1]]

         for layer in self.transformerlayers:
              embeddings = layer(embeddings, cosine, sine)

         embeddings = self.finalRMSnorm(embeddings)

         logits = self.outputprojection(embeddings)

         return logits



class GroupQueryAttention(nn.Module):
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args

         # im not entirely sure why qwen's dim in multi head is larger than the original dimension

        self.Wq = nn.Linear(self.model_args.dim, self.model_args.numheads * self.model_args.head_dim, bias = False)
        self.Wk = nn.Linear(self.model_args.dim, self.model_args.numkvheads * self.model_args.head_dim, bias = False)
        self.Wv = nn.Linear(self.model_args.dim, self.model_args.numkvheads * self.model_args.head_dim, bias = False)

        self.output = nn.Linear(self.model_args.numheads * self.model_args.head_dim, self.model_args.dim, bias = False)

        self.local_numheads = self.model_args.numheads
        self.local_numkvheads = self.model_args.numkvheads

        self.rmsNormQ = nn.RMSNorm(self.model_args.head_dim, eps = model_args.norm_eps)
        self.rmsNormK = nn.RMSNorm(self.model_args.head_dim, eps = model_args.norm_eps)

    def forward(self, tensor, cosine, sine):
        Q = self.Wq(tensor).reshape(tensor.shape[0], tensor.shape[1],self.local_numheads, self.model_args.head_dim)
        K = self.Wk(tensor).reshape(tensor.shape[0], tensor.shape[1], self.local_numkvheads, self.model_args.head_dim)
        V = self.Wv(tensor).reshape( tensor.shape[0], tensor.shape[1], self.local_numkvheads, self.model_args.head_dim)

        normalizedQ = self.rmsNormQ(Q)
        normalizedK = self.rmsNormK(K)

        rotatedQ = _RoPE(normalizedQ, cosine, sine, self.model_args.head_dim)
        rotatedK = _RoPE(normalizedK, cosine, sine, self.model_args.head_dim)

        headsRatio = self.model_args.numheads // self.model_args.numkvheads

        # we have more regular attention heads than KV heads under qwen's grouped query attention.
        # as a result, we need to keep all the dims lining up when we multiply

        # one way to do this is just repeat all of the values of KV so that the elementwise multiplication goes smoothly
        # we can also do some braodcsating shenanigans

        # do some wizardy on the dims to simply be able to matmul Q * K

        reshapedQ = rotatedQ.view(tensor.shape[0], tensor.shape[1], self.local_numkvheads, headsRatio, self.model_args.head_dim)
        reshapedQ = reshapedQ.permute(0, 2, 3, 1, 4)

        reshapedK = rotatedK.permute(0, 2, 3, 1).unsqueeze(2)
        reshapedV = V.permute(0,2,1,3).unsqueeze(2)

        if getattr(self, "cp_mode", False):
            # Ring attention expects keys with sequence before head dimension.
            keys = reshapedK.transpose(-2, -1).expand(-1, -1, headsRatio, -1, -1).contiguous()
            values = reshapedV.expand(-1, -1, headsRatio, -1, -1).contiguous()
            attention = ring_attention(reshapedQ, keys, values)
        else:
            attention = reshapedQ @ reshapedK
            attention = attention / (self.model_args.head_dim) ** 0.5

            # we have to do the manual causal masking and stuff now

            causal_mask = torch.full((tensor.shape[1], tensor.shape[1]), float("-inf"), device = attention.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)

            attention = attention + causal_mask

            attention = torch.softmax(attention, dim=-1, dtype=torch.float32).to(reshapedV.dtype) @ reshapedV

        attention = attention.permute(0, 3, 1, 2, 4)
        attention = attention.reshape(tensor.shape[0], tensor.shape[1], self.local_numheads * self.model_args.head_dim)

        return self.output(attention)

class Transformer(nn.Module):
    def __init__(self, model_args):
          super().__init__()

          self.model_args = model_args

          self.attention = GroupQueryAttention(model_args)
          self.moe = MoE(model_args)

          self.firstnorm = nn.RMSNorm(model_args.dim, eps = model_args.norm_eps)
          self.secondnorm = nn.RMSNorm(model_args.dim, eps = model_args.norm_eps)

          #whereas a standard transformer does normalization after some layer, Qwen3 does normalization before it,
          # "prenorm"

        
    def forward(self, tokens, cosine, sine):
         firstnorm = self.firstnorm(tokens)
         attention = self.attention(firstnorm, cosine, sine)

         residual = tokens + attention

         secondnorm = self.secondnorm(residual)
         ffn = self.moe(secondnorm) 

         residual2 = residual + ffn

         return residual2
