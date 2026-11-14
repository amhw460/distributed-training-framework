
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import Qwen3Model
from model_args import Qwen3ModelArgs
from train import Trainer
from data import TextDataset

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 2. Dataset
documents = [
    "Machine learning is used to train neural networks. " * 20,
    "Distributed training uses multiple GPUs to train models. " * 20,
    "Transformers use attention to process sequences of tokens. " * 20,
    "Mixture of experts models route tokens to specialized experts. " * 20,
]

dataset = TextDataset(documents, tokenizer, sequencelen=16)
loader = DataLoader(dataset, batch_size=4)

# 3. Small Qwen3-MoE model
args = Qwen3ModelArgs(
    vocab_size=len(tokenizer),
    max_seq_len=16,
    dim=32,
    moe_inter_dim=64,
    numlayers=2,
    numheads=4,
    numkvheads=2,
    head_dim=8,
    experts=4,
    k=2,
)

model = Qwen3Model(args)

# 4. Optimizer and Trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, "cpu")

# 5. Train
trainer.train(loader, 5)
