import argparse
import json
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from framework.data.cleaning import DocumentCleaner


# save a chunk of cleaned documents to a compressed parquet file
def write_shard(documents, output_dir, index):
    filename = f"part-{index:05d}.parquet"
    table = pa.table({"text": documents})
    pq.write_table(table, output_dir / filename, compression="zstd")
    return {"file": filename, "documents": len(documents)}


# clean documents and write them out in fixed-size shards
def prepare_documents(documents, output_dir, max_documents=100000, shard_size=10000, min_words=50, source="HuggingFaceFW/fineweb/sample-10BT/train", seed=42, shuffle_buffer=10000):
    if max_documents < 1 or shard_size < 1 or shuffle_buffer < 1:
        raise ValueError("max_documents, shard_size and shuffle_buffer must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cleaner = DocumentCleaner(min_words=min_words)
    buffer = []
    shards = []

    for document in islice(documents, max_documents):
        cleaned, reason = cleaner.clean(document["text"])
        if reason is not None:
            continue

        buffer.append(cleaned)
        if len(buffer) == shard_size:
            shards.append(write_shard(buffer, output_dir, len(shards)))
            buffer = []

    if buffer:
        shards.append(write_shard(buffer, output_dir, len(shards)))

    manifest = {
        "source": source,
        "requested_documents": max_documents,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "min_words": min_words,
        "shard_size": shard_size,
        "stats": cleaner.stats(),
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fineweb-100k"))
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    cfg = parser.parse_args()

    if cfg.max_documents < 1 or cfg.shard_size < 1 or cfg.shuffle_buffer < 1 or cfg.min_words < 0:
        parser.error("document count, shard size and shuffle buffer must be positive; min_words must be nonnegative")

    # stream from FineWeb without downloading the full dataset
    from datasets import load_dataset
    documents = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    shuffle_buffer = min(cfg.shuffle_buffer, cfg.max_documents)
    documents = documents.shuffle(seed=cfg.seed, buffer_size=shuffle_buffer)

    manifest = prepare_documents(documents, cfg.output_dir, max_documents=cfg.max_documents, shard_size=cfg.shard_size, min_words=cfg.min_words, seed=cfg.seed, shuffle_buffer=shuffle_buffer)
    print(f"Processed {manifest['stats']['processed']} documents, kept {manifest['stats']['kept']}, wrote {len(manifest['shards'])} shards to {cfg.output_dir}")
    print(f"Rejections: empty={manifest['stats']['empty']}, too_short={manifest['stats']['too_short']}, duplicate={manifest['stats']['duplicate']}")


if __name__ == "__main__":
    main()
