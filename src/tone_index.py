"""Turn the fine-tuned classifier into a tone index.

Put one statement per file in statements/, named by date, e.g.
  statements/2025-01-29_fed.txt
  statements/2025-01-29_boc.txt
Each document is split into sentences, every sentence is classified, and

  tone = (hawkish - dovish) / number of sentences      (range -1 to +1)

Output: results/tone_index.csv, results/tone_index.png,
        results/tone_sentences.csv (every sentence with its label).

Usage:
  python src/tone_index.py --adapter adapters
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mlx_lm import load

from evaluate import classify

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]   # drop fragments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="adapters")
    ap.add_argument("--folder", default="statements")
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files in {args.folder}/")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    doc_rows, sent_rows = [], []
    for f in files:
        m = DATE_RE.search(f.stem)
        if not m:
            print("Skipping (no YYYY-MM-DD in name):", f.name)
            continue
        source = f.stem.replace(m.group(1), "").strip("_- ") or "doc"
        sents = split_sentences(f.read_text(errors="ignore"))
        labels = [classify(model, tokenizer, s)[1] for s in sents]
        for s, l in zip(sents, labels):
            sent_rows.append({"date": m.group(1), "source": source, "label": l, "sentence": s})
        n = len(sents)
        h, d = labels.count("hawkish"), labels.count("dovish")
        doc_rows.append({"date": m.group(1), "source": source, "sentences": n,
                         "hawkish": h, "dovish": d, "neutral": labels.count("neutral"),
                         "tone": round((h - d) / n, 4) if n else None})
        print(f"{f.name}: {n} sentences, tone={doc_rows[-1]['tone']}")

    df = pd.DataFrame(doc_rows).sort_values(["source", "date"])
    df["date"] = pd.to_datetime(df["date"])
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/tone_index.csv", index=False)
    pd.DataFrame(sent_rows).to_csv("results/tone_sentences.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for src, g in df.groupby("source"):
        ax.plot(g["date"], g["tone"], marker="o", label=src)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("tone  (hawkish − dovish) / sentences")
    ax.set_title("Central bank tone index")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("results/tone_index.png", dpi=150)
    print("Saved results/tone_index.csv and results/tone_index.png")


if __name__ == "__main__":
    main()
