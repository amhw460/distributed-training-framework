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


# render a basic histogram with axes
def chart(title, values, bounds):
    bars = histogram(values, bounds)
    width, height = 800, 260
    left, right, top, bottom = 56, 12, 16, 48
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max((count for _, count in bars), default=0) or 1
    step = plot_width / len(bars)

    result = [f"<section><h2>{html.escape(title)}</h2><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"]
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_height * (1 - tick / 4)
        result.append(f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}'/>")
        result.append(f"<text class='axis-label' x='{left - 8}' y='{y + 4:.1f}' text-anchor='end'>{value:,.0f}</text>")

    for i, (label, count) in enumerate(bars):
        x = left + (i + 0.1) * step
        bar_height = plot_height * count / maximum
        y = top + plot_height - bar_height
        result.append(f"<rect class='bar' x='{x:.1f}' y='{y:.1f}' width='{step * 0.8:.1f}' height='{bar_height:.1f}'/>")
        result.append(f"<text class='axis-label' x='{x + step * 0.4:.1f}' y='{height - 22}' text-anchor='middle'>{html.escape(label)}</text>")

    result.append("</svg></section>")
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

    summary = [
        ("Documents", f"{total:,}"),
        ("Average words", f"{average_words:,.0f}"),
        ("Median words", f"{median_words:,}"),
        ("Tokenized sample", f"{len(examples):,}"),
        ("Average sample tokens", f"{average_tokens:,.0f}"),
    ]
    summary_html = "\n".join(f"<tr><th>{html.escape(label)}</th><td><strong>{value}</strong></td></tr>" for label, value in summary)
    cleaning_html = "\n".join(f"<tr><th>{html.escape(label)}</th><td>{cleaning[key]:,}</td></tr>" for key, label in [("processed", "Processed"), ("kept", "Kept"), ("empty", "Empty"), ("too_short", "Too short"), ("duplicate", "Duplicate")] if key in cleaning)
    cleaning_html = f"<table>{cleaning_html}</table>" if cleaning_html else "<p>No cleaning manifest was found.</p>"

    # escape document text before adding it to HTML
    previews = []
    for index, text in examples[:preview_count]:
        excerpt = text[:600] + ("…" if len(text) > 600 else "")
        previews.append(f"<details><summary>Document #{index + 1:,}</summary><pre>{html.escape(excerpt)}</pre></details>")


    doc_chart = chart("Document lengths (words, all documents)", word_lengths, WORD_BINS)
    token_chart = chart(f"Token lengths (sample of {len(token_lengths):,} documents)", token_lengths, TOKEN_BINS)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset inspection</title>
<style>
body {{ font-family: Arial, sans-serif; color: #222; background: white; margin: 0; }}
main {{ max-width: 900px; margin: 36px auto; padding: 0 20px 40px; }}
h1 {{ font-size: 24px; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 30px 0 12px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 420px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px 12px 7px 0; }}
th {{ text-align: left; font-weight: normal; }}
td {{ text-align: right; font-variant-numeric: tabular-nums; }}
svg {{ display: block; width: 100%; height: auto; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
.bar {{ fill: #777; }}
.axis-label {{ font-family: Arial, sans-serif; font-size: 11px; fill: #444; }}
.subtitle, .note {{ font-size: 13px; color: #555; }}
details {{ border-bottom: 1px solid #ddd; padding: 10px 0; }}
pre {{ font-size: 13px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<main>
<h1>Dataset inspection</h1>
<p class="subtitle">{html.escape(source)} · {len(files)} Parquet shards</p>
<h2>Summary</h2>
<table>{summary_html}</table>
{doc_chart}
{token_chart}
<p class="note">Document lengths use all saved documents. Token lengths use a random sample of {len(examples):,} documents (seed {seed}).</p>
<h2>Cleaning</h2>
{cleaning_html}
<h2>Sample documents</h2>
{"".join(previews) if previews else "<p>No samples available.</p>"}
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
