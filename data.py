from transformers import AutoTokenizer
import torch
from torch.utils.data import IterableDataset, DataLoader

# im just using the qwen native tokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# test = "hello how are u"
# tokens = tokenizer.encode(test)
# print(tokens)

# print(tokenizer.decode(tokens))

from datasets import load_dataset

def build_fineweb_dataset(tokenizer, sequencelen):
    documents = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    return TextDataset(documents, tokenizer, sequencelen)

class TextDataset(IterableDataset):
  def __init__(self, documents, tokenizer, sequencelen):
    
    self.documents = documents
    self.tokenizer = tokenizer
    self.sequencelen = sequencelen

    self.buffer = []
    self.samples_seen = 0
  
  def __iter__(self):
    # how do we produce trainign examples 
    buffer = []
    samplestoskip = self.samples_seen

    for document in self.documents:
      text = document["text"] if isinstance(document, dict) else document
      tokens = self.tokenizer.encode(text, add_special_tokens=False)
      tokens.append(self.tokenizer.eos_token_id)
      buffer.extend(tokens)

      while len(buffer) >= self.sequencelen + 1:

        first = buffer[:self.sequencelen + 1]
        buffer = buffer[self.sequencelen + 1:]

        if samplestoskip > 0:
          samplestoskip -= 1
          continue
        
        self.samples_seen += 1

        yield torch.tensor(first, dtype=torch.long)

  def state(self):
    return self.samples_seen

  def load_state(self, state):
    self.samples_seen = state

