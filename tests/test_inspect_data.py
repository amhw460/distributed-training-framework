import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.inspect_data import histogram, inspect_dataset


class FakeTokenizer:
    # tokenize each word into one fake token without network access
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class InspectDataTests(unittest.TestCase):

    # create shards with known document lengths and a manifest
    def create_dataset(self, folder):
        texts = ["one two three", "hello <script>alert(1)</script> world", "a " * 75, "b " * 120, "c " * 220, "d " * 400]
        pq.write_table(pa.table({"text": texts[:3]}), folder / "part-00000.parquet")
        pq.write_table(pa.table({"text": texts[3:]}), folder / "part-00001.parquet")
        (folder / "manifest.json").write_text(json.dumps({"source": "test source", "stats": {"processed": 8, "kept": 6, "empty": 1, "too_short": 1, "duplicate": 0}}))
        return texts

    def test_histogram(self):
        self.assertEqual(histogram([0, 2, 4, 10], [2, 5]), [("0–1", 1), ("2–4", 2), ("5+", 1)])

    # verify the HTML contains the document statistics, charts and cleaning results
    def test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            result = inspect_dataset(folder, folder / "report.html", FakeTokenizer(), sample_size=100, preview_count=6)
            report = (folder / "report.html").read_text()
            self.assertEqual(result["documents"], 6)
            self.assertEqual(result["sampled"], 6)
            self.assertIn("Document lengths (words, all documents)", report)
            self.assertIn("Token lengths (sample of 6 documents)", report)
            self.assertIn("Too short", report)
            self.assertIn("<strong>6</strong>", report)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
            self.assertNotIn("<script>alert(1)</script>", report)

    # sampling must stay reproducible for the same random seed
    def test_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.create_dataset(folder)
            first = inspect_dataset(folder, folder / "first.html", FakeTokenizer(), sample_size=2, preview_count=2, seed=3)
            second = inspect_dataset(folder, folder / "second.html", FakeTokenizer(), sample_size=2, preview_count=2, seed=3)
            self.assertEqual(first["sampled"], 2)
            self.assertEqual(first["average_tokens"], second["average_tokens"])
            self.assertEqual((folder / "first.html").read_text(), (folder / "second.html").read_text())

    # report can still be created for an arbitrary parquet directory without a manifest
    def test_optional_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pq.write_table(pa.table({"text": ["hello world"]}), folder / "part-00000.parquet")
            inspect_dataset(folder, folder / "report.html", FakeTokenizer())
            self.assertIn("No cleaning manifest was found.", (folder / "report.html").read_text())

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(ValueError):
                inspect_dataset(folder, folder / "report.html", FakeTokenizer())
            self.create_dataset(folder)
            with self.assertRaises(ValueError):
                inspect_dataset(folder, folder / "report.html", FakeTokenizer(), sample_size=0)
            (folder / "manifest.json").write_text(json.dumps({"stats": {"kept": 9}}))
            with self.assertRaises(ValueError):
                inspect_dataset(folder, folder / "report.html", FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
