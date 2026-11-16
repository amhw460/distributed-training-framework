import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.prepare_data import prepare_documents


class PrepareDataTests(unittest.TestCase):

    # verify cleaning counts, document contents, shard sizes and manifest
    def test_prepare_100_documents(self):
        texts = [f"document {i} has enough unique words to pass the filter" for i in range(90)]
        documents = [{"text": text} for text in texts]
        documents += [{"text": text} for text in texts[:3]]
        documents += [{"text": ""}, {"text": "!!"}]
        documents += [{"text": "short"} for _ in range(5)]
        self.assertEqual(len(documents), 100)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed"
            manifest = prepare_documents(documents, output, max_documents=100, shard_size=25, min_words=5, source="test", seed=42, shuffle_buffer=100)

            self.assertEqual(manifest["stats"], {"processed": 100, "kept": 90, "empty": 2, "too_short": 5, "duplicate": 3})
            self.assertEqual([shard["documents"] for shard in manifest["shards"]], [25, 25, 25, 15])
            self.assertEqual(manifest["source"], "test")
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)

            saved = []
            for shard in manifest["shards"]:
                saved.extend(pq.read_table(output / shard["file"]).column("text").to_pylist())
            self.assertEqual(saved, texts)

    # stop after the requested number of raw documents, not the number kept
    def test_max_documents_limits_input(self):
        documents = [{"text": "short"} for _ in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed"
            manifest = prepare_documents(iter(documents), output, max_documents=3, min_words=50)
            self.assertEqual(manifest["stats"]["processed"], 3)
            self.assertEqual(manifest["stats"]["kept"], 0)
            self.assertEqual(manifest["shards"], [])

    # prevent accidental overwriting of an existing dataset
    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(FileExistsError):
                prepare_documents([], output, max_documents=1)

    def test_invalid_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prepare_documents([], Path(directory) / "processed", shard_size=0)


if __name__ == "__main__":
    unittest.main()
