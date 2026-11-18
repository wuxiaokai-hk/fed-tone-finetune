"""Shared prompt template, labels and metrics.

The SAME prompt is used for training and for evaluation, so the
fine-tuned model is tested exactly as it was taught.
"""

LABELS = ["dovish", "hawkish", "neutral"]          # dataset: 0, 1, 2
LABEL_FROM_INT = {0: "dovish", 1: "hawkish", 2: "neutral"}

INSTRUCTION = (
    "Classify the monetary-policy stance of this central bank sentence.\n"
    "hawkish = points to tighter policy (higher rates, inflation worries).\n"
    "dovish = points to easier policy (lower rates, weak growth or jobs).\n"
    "neutral = neither.\n"
    "Answer with one word: hawkish, dovish or neutral.\n\n"
    "Sentence: {sentence}"
)

FEW_SHOT = [
    ("Inflation remains elevated and further policy firming may be appropriate.", "hawkish"),
    ("Downside risks to employment have increased, warranting additional accommodation.", "dovish"),
    ("Broad equity price indexes fell over the intermeeting period.", "neutral"),
]


def user_prompt(sentence: str, few_shot: bool = False) -> str:
    text = INSTRUCTION.format(sentence=sentence.strip())
    if few_shot:
        examples = "\n".join(f"Sentence: {s}\nAnswer: {a}" for s, a in FEW_SHOT)
        text = "Examples:\n" + examples + "\n\n" + text
    return text


def parse_label(output: str) -> str:
    """Return the first label word found in the model output, else 'unknown'."""
    out = output.lower()
    hits = [(out.find(l), l) for l in LABELS if l in out]
    return min(hits)[1] if hits else "unknown"


def metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1 and a confusion matrix, with no extra libraries."""
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    f1s = {}
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    confusion = {t: {p: 0 for p in LABELS + ["unknown"]} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s.values()) / len(LABELS), 4),
        "f1_by_class": {k: round(v, 4) for k, v in f1s.items()},
        "unparseable": sum(p == "unknown" for p in y_pred),
        "confusion": confusion,
    }
