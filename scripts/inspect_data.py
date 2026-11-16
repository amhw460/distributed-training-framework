import argparse
import html
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


WORD_BINS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
TOKEN_BINS = [128, 256, 512, 1024, 2048, 4096, 8192]


# count how many documents fall into each length range
def histogram(values, bounds):
    counts = [0] * (len(bounds) + 1)
    for value in values:
        bucket = next((i for i, bound in enumerate(bounds) if value < bound), len(bounds))
        counts[bucket] += 1

    labels = [f"0–{bounds[0] - 1}"]
    labels += [f"{bounds[i - 1]}–{bounds[i] - 1}" for i in range(1, len(bounds))]
    labels.append(f"{bounds[-1]}+")
    return list(zip(labels, counts))


# render a labeled bar chart without any external plotting dependencies
def chart(title, values, bounds):
    bars = histogram(values, bounds)
    largest = max((count for _, count in bars), default=0)
    result = [f"<section class='panel'><h2>{html.escape(title)}</h2>"]
    for label, count in bars:
        width = 100 * count / largest if largest else 0
        result.append(f"<div class='bar-row'><span class='range'>{html.escape(label)}</span><div class='track'><div class='fill' style='width:{width:.2f}%'></div></div><span class='count'>{count:,}</span></div>")
    result.append("</section>")
    return "\n".join(result)


# inspect all document lengths, but tokenize only a random bounded-size sample
def inspect_dataset(dataset_path, output_path, tokenizer, sample_size=1000, preview_count=5, seed=42):
    if sample_size < 1 or preview_count < 0:
        raise ValueError("sample_size must be positive and preview_count must be nonnegative")

    directory = Path(dataset_path)
    files = sorted(directory.glob("part-*.parquet"))
    if not files:
        raise ValueError(f"No part-*.parquet files found in {directory}")

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rng = random.Random(seed)
    word_lengths = []
    examples = []

    # reservoir sampling gives every document the same chance of being inspected
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column(0).to_pylist():
                index = len(word_lengths)
                word_lengths.append(len(text.split()))
                if len(examples) < sample_size:
                    examples.append((index, text))
                else:
                    selected = rng.randrange(index + 1)
                    if selected < sample_size:
                        examples[selected] = (index, text)

    total = len(word_lengths)
    kept = manifest.get("stats", {}).get("kept")
    if kept is not None and kept != total:
        raise ValueError(f"Manifest says {kept} documents, but Parquet contains {total}")

    examples.sort(key=lambda item: item[0])
    token_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for _, text in examples]
    sorted_words = sorted(word_lengths)
    median_words = sorted_words[(total - 1) // 2] if total else 0
    average_words = sum(word_lengths) / total if total else 0
    average_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    source = manifest.get("source", "Local Parquet dataset")
    cleaning = manifest.get("stats", {})

    cards = [("Documents", f"{total:,}"), ("Average words", f"{average_words:,.0f}"), ("Median words", f"{median_words:,}"), ("Tokenized sample", f"{len(examples):,}")]
    card_html = "\n".join(f"<div class='stat'><strong>{value}</strong><span>{label}</span></div>" for label, value in cards)
    cleaning_html = "\n".join(f"<div class='metric'><span>{html.escape(label)}</span><strong>{cleaning[key]:,}</strong></div>" for key, label in [("processed", "Processed"), ("kept", "Kept"), ("empty", "Empty"), ("too_short", "Too short"), ("duplicate", "Duplicate")] if key in cleaning)
    if not cleaning_html:
        cleaning_html = "<p>No cleaning manifest was found.</p>"

    # escape document text before adding it to HTML
    previews = []
    for index, text in examples[:preview_count]:
        excerpt = text[:600] + ("…" if len(text) > 600 else "")
        previews.append(f"<article class='preview'><div class='preview-label'>Document #{index + 1:,}</div><pre>{html.escape(excerpt)}</pre></article>")

    doc_chart = chart("Document lengths (words, all documents)", word_lengths, WORD_BINS)
    token_chart = chart(f"Token lengths (sample of {len(token_lengths):,} documents)", token_lengths, TOKEN_BINS)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset inspection</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #171923; margin: 0; }}
main {{ max-width: 1000px; margin: 48px auto; padding: 0 24px 64px; }}
h1 {{ font-size: 32px; letter-spacing: -.03em; margin-bottom: 8px; }}
h2 {{ font-size: 19px; margin: 0 0 20px; }}
.subtitle {{ color: #626977; margin: 0 0 28px; overflow-wrap: anywhere; }}
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat, .panel {{ background: white; border: 1px solid #e3e5ea; border-radius: 12px; padding: 24px; }}
.stat strong {{ font-size: 28px; display: block; }}
.stat span, .preview-label {{ color: #646b79; font-size: 13px; }}
.panels {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
.bar-row {{ display: grid; grid-template-columns: 82px minmax(0, 1fr) 60px; align-items: center; gap: 10px; margin: 12px 0; font-size: 12px; }}
.track {{ height: 17px; border-radius: 4px; background: #eff1f5; overflow: hidden; }}
.fill {{ height: 100%; background: #4f74c8; border-radius: 4px; }}
.count {{ text-align: right; font-variant-numeric: tabular-nums; }}
.metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eceef2; }}
.metric:last-child {{ border-bottom: 0; }}
.metric strong {{ font-variant-numeric: tabular-nums; }}
.preview {{ padding: 14px 0; border-top: 1px solid #eceef2; }}
.preview pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 220px; overflow: auto; font-size: 13px; line-height: 1.55; }}
.note {{ color: #626977; font-size: 13px; line-height: 1.5; margin: 16px 0 24px; }}
@media (max-width: 760px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .panels {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>Dataset inspection</h1>
<p class="subtitle">{html.escape(source)} · {len(files)} Parquet shards · seed {seed}</p>
<div class="stats">{card_html}</div>
<div class="panels">{doc_chart}{token_chart}</div>
<p class="note">Word lengths are measured over all saved documents. Token lengths use a reproducible random sample of {len(examples):,} documents, tokenized without special tokens. Counts in the two charts therefore have different denominators. Average sampled token length: {average_tokens:,.0f}.</p>
<section class="panel"><h2>Cleaning results</h2>{cleaning_html}</section>
<section class="panel" style="margin-top:20px"><h2>Sample documents</h2>{"".join(previews) if previews else "<p>No samples available.</p>"}</section>
</main>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return {"documents": total, "sampled": len(examples), "average_words": average_words, "average_tokens": average_tokens, "output": str(output_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=Path("data/fineweb-100k"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--previews", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    cfg = parser.parse_args()

    # only load the tokenizer for inspection, not when importing this script for tests
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    output = cfg.output or cfg.dataset_path / "report.html"
    result = inspect_dataset(cfg.dataset_path, output, tokenizer, sample_size=cfg.sample_size, preview_count=cfg.previews, seed=cfg.seed)
    print(f"Inspected {result['documents']:,} documents, tokenized {result['sampled']:,} samples, saved {result['output']}")


if __name__ == "__main__":
    main()
