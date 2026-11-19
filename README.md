# fed-tone-finetune

Fine-tune a small open LLM (Qwen2.5-1.5B) with LoRA so it labels central-bank
sentences as **hawkish**, **dovish** or **neutral**, then measure how much that
training improves on prompting alone. As an extension, turn the model into a
tone index for Fed / Bank of Canada statements.

Runs locally on an Apple Silicon Mac (M1 or later, 16 GB RAM recommended) using
Apple's MLX. No GPU and no API key needed.

## The question

> How much does fine-tuning add beyond a good prompt?

| Run | What changes | Weights updated? |
|---|---|---|
| zero_shot | instruction only | no |
| few_shot | instruction + 3 examples | no |
| finetuned | LoRA trained on ~1,700 labelled sentences | yes (small adapter) |

All three are scored on the same held-out test set.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the model (~3 GB) and the dataset from Hugging Face.

## Run everything

```bash
bash run_all.sh            # about 30–60 min on an M-series Mac
ITERS=200 bash run_all.sh  # quicker first pass
```

Or step by step:

```bash
python src/prepare_data.py                          # data/*.jsonl
python src/evaluate.py --name zero_shot --limit 100 # smoke test
python src/evaluate.py --name zero_shot
python src/evaluate.py --name few_shot --few-shot
mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data data \
  --iters 600 --batch-size 4 --num-layers 16 --learning-rate 1e-5 \
  --steps-per-eval 50 --mask-prompt --adapter-path adapters 2>&1 | tee logs/train.log
python src/evaluate.py --name finetuned --adapter adapters
python src/compare.py                               # results/summary.md + loss_curve.png
```

## Outputs

- `results/summary.md`: accuracy and macro-F1 for each run
- `results/<run>.json`: full metrics and a confusion matrix
- `results/<run>_predictions.jsonl`: every test sentence with its prediction (good for error analysis)
- `results/loss_curve.png`: train vs validation loss
- `adapters/`: the trained LoRA adapter (the "product")

## What the pieces do

- `src/common.py`: one prompt template shared by training and evaluation, plus metrics
- `src/prepare_data.py`: downloads the data and converts it to chat-format JSONL
- `src/evaluate.py`: greedy decoding, parses the first label word, scores it
- `src/compare.py`: results table and loss plot
- `src/tone_index.py`: the extension (below)

`--mask-prompt` means the loss is computed only on the answer word, not the
instruction, so the model learns the classification rather than how to repeat
the prompt.

## Extension: tone index

1. Save statements as plain text in `statements/`, with the date in the name:
   `2025-01-29_fed.txt`, `2025-01-29_boc.txt`, …
2. `python src/tone_index.py --adapter adapters`

For each document: tone = (hawkish − dovish) / sentences. Output goes to
`results/tone_index.csv`, `results/tone_index.png` and
`results/tone_sentences.csv`.

Bank of Canada statements are an out-of-sample test: the model was trained only
on FOMC text.

## Things to try next

- Vary `--iters` (100, 300, 600, 1200) and watch where validation loss turns up (overfitting).
- Vary the training size (25%, 50%, 100%) and plot F1 against it: a learning curve.
- Swap the base model (`MODEL=meta-llama/Llama-3.2-1B-Instruct bash run_all.sh`; Llama models need a Hugging Face login).
- Merge the adapter into one model: `mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path adapters --save-path fused`

## If something breaks

- `unrecognized arguments: --mask-prompt` or `--num-layers`: your mlx-lm is old. Run `pip install -U mlx-lm`.
- Out of memory: use `--batch-size 2`, or add `--grad-checkpoint`.
- Dataset not found or column error: check the name on Hugging Face, then pass `--dataset <name>` to `prepare_data.py`. The script accepts a `sentence` or `text` column and integer or string labels.

## Data and licence

The data comes from Shah, Paturi & Chava (2023), "Trillion Dollar Words: A New
Financial Dataset, Task & Market Analysis" (ACL), released under CC BY-NC 4.0:
non-commercial use only, so cite the paper. Labels: 0 = dovish, 1 = hawkish,
2 = neutral.

## License

MIT, see [LICENSE](LICENSE). Free to use, change and share. Data downloaded from public sources stay under their providers' terms. Independent open-source project, not affiliated with or endorsed by any employer, regulator or data vendor.
