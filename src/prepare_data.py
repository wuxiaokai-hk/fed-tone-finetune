"""Download the FOMC hawkish/dovish dataset and write chat-format JSONL.

Output (what mlx_lm.lora expects):
  data/train.jsonl, data/valid.jsonl, data/test.jsonl
Each line: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  python src/prepare_data.py
  python src/prepare_data.py --dataset gtfintechlab/fomc_communication --valid-frac 0.1
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from common import LABEL_FROM_INT, LABELS, user_prompt

TEXT_COLS = ["sentence", "text", "Sentence"]


def to_label(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in LABELS:
            return v
        value = int(v)  # "0", "1", "2"
    return LABEL_FROM_INT[int(value)]


def to_records(rows):
    cols = rows[0].keys()
    text_col = next(c for c in TEXT_COLS if c in cols)
    out = []
    for r in rows:
        sentence = (r[text_col] or "").strip()
        if not sentence:
            continue
        out.append({"sentence": sentence, "label": to_label(r["label"])})
    return out


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            msg = {"messages": [
                {"role": "user", "content": user_prompt(r["sentence"])},
                {"role": "assistant", "content": r["label"]},
            ]}
            f.write(json.dumps(msg) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtfintechlab/fomc_communication")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    print("Splits found:", {k: len(v) for k, v in ds.items()})

    train = to_records(list(ds["train"]))
    if "test" in ds:
        test = to_records(list(ds["test"]))
    else:  # no official test split: hold out 20%
        random.Random(args.seed).shuffle(train)
        cut = int(len(train) * 0.2)
        test, train = train[:cut], train[cut:]

    # carve a validation set out of train (used to watch for overfitting)
    random.Random(args.seed).shuffle(train)
    cut = int(len(train) * args.valid_frac)
    valid, train = train[:cut], train[cut:]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, recs in [("train", train), ("valid", valid), ("test", test)]:
        write_jsonl(out / f"{name}.jsonl", recs)
        counts = {l: sum(r["label"] == l for r in recs) for l in LABELS}
        print(f"{name:5s}: {len(recs):5d} rows  {counts}")


if __name__ == "__main__":
    main()
