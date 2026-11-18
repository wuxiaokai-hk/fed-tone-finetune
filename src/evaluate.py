"""Score a model on data/test.jsonl and save metrics + predictions.

Three runs give the full comparison:
  python src/evaluate.py --name zero_shot
  python src/evaluate.py --name few_shot  --few-shot
  python src/evaluate.py --name finetuned --adapter adapters

Add --limit 100 for a quick smoke test.
"""
import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load

from common import metrics, parse_label, user_prompt

SENTENCE_RE = re.compile(r"Sentence: (.*)\Z", re.S)


def load_test(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            m = json.loads(line)["messages"]
            sentence = SENTENCE_RE.search(m[0]["content"]).group(1)
            rows.append((sentence, m[1]["content"]))
    return rows[:limit] if limit else rows


def classify(model, tokenizer, sentence, few_shot=False):
    msgs = [{"role": "user", "content": user_prompt(sentence, few_shot)}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    return out, parse_label(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="folder from mlx_lm.lora")
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = load_test(args.test, args.limit)

    preds, records, t0 = [], [], time.time()
    for i, (sentence, gold) in enumerate(rows, 1):
        raw, pred = classify(model, tokenizer, sentence, args.few_shot)
        preds.append(pred)
        records.append({"sentence": sentence, "gold": gold, "pred": pred, "raw": raw})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)")

    res = metrics([g for _, g in rows], preds)
    res.update(name=args.name, model=args.model, adapter=args.adapter, few_shot=args.few_shot)

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=2))
    with open(out / f"{args.name}_predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: accuracy={res['accuracy']}  macro_f1={res['macro_f1']}  "
          f"unparseable={res['unparseable']}")


if __name__ == "__main__":
    main()
