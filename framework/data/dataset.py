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

def build_fineweb_dataset(tokenizer, sequencelen, rank=0, world_size=1):
    documents = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    if world_size > 1:
        documents = documents.shard(
            num_shards=world_size,
            index=rank,
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



# read only parquet shards, leaving the manifest and other files alone
class ParquetDocumentDataset:
  def __init__(self, directory, rank=0, world_size=1, batch_size=1024):
    from pathlib import Path
    self.directory = Path(directory)
    self.files = sorted(self.directory.glob("part-*.parquet"))
    if not self.files:
      raise ValueError(f"No part-*.parquet files found in {self.directory}")
    if world_size < 1 or not 0 <= rank < world_size:
      raise ValueError("Invalid data replica rank or world size")
    if batch_size < 1:
      raise ValueError("batch_size must be positive")

    self.rank = rank
    self.world_size = world_size
    self.batch_size = batch_size

  # stream the text column in batches and give each replica different documents
  def __iter__(self):
    import pyarrow.parquet as pq

    index = 0
    for path in self.files:
      parquet = pq.ParquetFile(path)
      for batch in parquet.iter_batches(batch_size=self.batch_size, columns=["text"]):
        for text in batch.column(0).to_pylist():
          if index % self.world_size == self.rank:
            yield {"text": text}
          index += 1


# reuse the existing tokenization and sequence packing for local parquet data
def build_parquet_dataset(directory, tokenizer, sequencelen, rank=0, world_size=1):
  documents = ParquetDocumentDataset(directory, rank=rank, world_size=world_size)
  return TextDataset(documents, tokenizer, sequencelen)
