#!/usr/bin/env bash
# Full pipeline: data -> baselines -> LoRA training -> fine-tuned eval -> summary.
# Run from the repo root:  bash run_all.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ITERS="${ITERS:-600}"
mkdir -p logs results

echo "== 1. Prepare data"
python src/prepare_data.py

echo "== 2. Baselines (no training)"
python src/evaluate.py --model "$MODEL" --name zero_shot
python src/evaluate.py --model "$MODEL" --name few_shot --few-shot

echo "== 3. LoRA fine-tuning"
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data data \
  --iters "$ITERS" \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --mask-prompt \
  --adapter-path adapters 2>&1 | tee logs/train.log

echo "== 4. Evaluate fine-tuned model"
python src/evaluate.py --model "$MODEL" --adapter adapters --name finetuned

echo "== 5. Summary"
python src/compare.py
