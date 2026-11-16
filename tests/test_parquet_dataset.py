import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader


# avoid downloading the Qwen tokenizer just to test the local parquet reader
with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None))}):
    from framework.data.dataset import ParquetDocumentDataset, build_parquet_dataset


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 90 for char in text]


class ParquetDatasetTests(unittest.TestCase):

    # write two parquet shards plus a manifest like prepare_data.py produces
    def create_dataset(self, folder):
        pq.write_table(pa.table({"text": ["abc", "def", "ghi"]}), folder / "part-00000.parquet")
        pq.write_table(pa.table({"text": ["jkl", "mno", "pqr", "stu"]}), folder / "part-00001.parquet")
        (folder / "manifest.json").write_text(json.dumps({"stats": {"kept": 7}}))

    def test_read_all_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            documents = list(ParquetDocumentDataset(folder, batch_size=2))
            self.assertEqual([row["text"] for row in documents], ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu"])

    # different replicas should never receive the same document
    def test_distributed_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            all_docs = [row["text"] for row in ParquetDocumentDataset(folder)]
            replicas = [[row["text"] for row in ParquetDocumentDataset(folder, rank=rank, world_size=3, batch_size=2)] for rank in range(3)]
            self.assertEqual(replicas[0], all_docs[0::3])
            self.assertEqual(replicas[1], all_docs[1::3])
            self.assertEqual(replicas[2], all_docs[2::3])
            self.assertEqual(sum(len(replica) for replica in replicas), len(all_docs))

    # reuse TextDataset to append EOS, pack tokens and batch them for the model
    def test_token_packing_and_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            dataset = build_parquet_dataset(folder, FakeTokenizer(), sequencelen=3)
            batches = list(DataLoader(dataset, batch_size=2, drop_last=True, num_workers=0))
            self.assertEqual(len(batches), 3)
            self.assertEqual(tuple(batches[0].shape), (2, 4))
            expected = [ord(c) % 90 for c in "abc"] + [99] + [ord(c) % 90 for c in "def"] + [99]
            self.assertEqual(batches[0].tolist(), [expected[:4], expected[4:8]])

    # nonexistent data, bad ranks and bad batch sizes should fail immediately
    def test_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(ValueError):
                ParquetDocumentDataset(folder)
            self.create_dataset(folder)
            with self.assertRaises(ValueError):
                ParquetDocumentDataset(folder, rank=2, world_size=2)
            with self.assertRaises(ValueError):
                ParquetDocumentDataset(folder, batch_size=0)

    # resume uses the existing packed sequence skip counter
    def test_resume_sequence_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            dataset = build_parquet_dataset(folder, FakeTokenizer(), sequencelen=3)
            original = list(dataset)
            dataset.samples_seen = 2
            resumed = list(dataset)
            self.assertEqual([tensor.tolist() for tensor in resumed], [tensor.tolist() for tensor in original[2:]])


if __name__ == "__main__":
    unittest.main()
