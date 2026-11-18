"""Build results/summary.md from every results/<name>.json, and plot the
training/validation loss from logs/train.log if it exists."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["zero_shot", "few_shot", "finetuned"]


def summary_table():
    files = {p.stem: p for p in Path("results").glob("*.json")}
    names = [n for n in ORDER if n in files] + sorted(set(files) - set(ORDER))
    lines = ["| Run | Accuracy | Macro-F1 | F1 dovish | F1 hawkish | F1 neutral | Unparseable |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        r = json.loads(files[n].read_text())
        f = r["f1_by_class"]
        lines.append(f"| {n} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {f['dovish']:.3f} | "
                     f"{f['hawkish']:.3f} | {f['neutral']:.3f} | {r['unparseable']} |")
    text = "\n".join(lines)
    Path("results/summary.md").write_text(text + "\n")
    print(text)


def loss_plot(log="logs/train.log"):
    if not Path(log).exists():
        return
    txt = Path(log).read_text()
    tr = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Train loss ([\d.]+)", txt)]
    va = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Val loss ([\d.]+)", txt)]
    if not tr and not va:
        print("No loss lines found in", log)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train")
    if va:
        ax.plot(*zip(*va), marker="o", label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("LoRA fine-tuning loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/loss_curve.png", dpi=150)
    print("Saved results/loss_curve.png")


if __name__ == "__main__":
    summary_table()
    loss_plot()
