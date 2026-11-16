import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


# qwen's tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


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
