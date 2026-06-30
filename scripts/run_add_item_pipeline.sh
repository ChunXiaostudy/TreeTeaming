#!/bin/bash
# add_item 全流程：准备 base -> Qwen 插入水果篮 -> 闭源推理 -> GPT Judge 打分
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOVE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOVE_ROOT"

FIGSTEP_SRC="${FIGSTEP_SRC:-}"
MMSAFETY_SRC="${MMSAFETY_SRC:-}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"
SKIP_GENERATE="${SKIP_GENERATE:-false}"
SKIP_INFER="${SKIP_INFER:-false}"
SKIP_SCORE="${SKIP_SCORE:-false}"
MODELS="${MODELS:-gpt-4o-2024-05-13}"
GPU_ID="${GPU_ID:-0}"

echo "=== Meta-Attack add_item pipeline ==="

if [[ "$SKIP_PREPARE" != "true" ]]; then
  if [[ -z "$FIGSTEP_SRC" || -z "$MMSAFETY_SRC" ]]; then
    echo "[prepare] skipped (set FIGSTEP_SRC and MMSAFETY_SRC to run prepare_base_images.py)"
  else
    echo "[1/4] Prepare base images..."
    python -u generate/prepare_base_images.py \
      --figstep-src "$FIGSTEP_SRC" \
      --mmsafety-src "$MMSAFETY_SRC"
  fi
else
  echo "[1/4] Prepare skipped (SKIP_PREPARE=true)"
fi

if [[ "$SKIP_GENERATE" != "true" ]]; then
  echo "[2/4] Generate add_item images (Qwen-Image-Edit)..."
  GPU_ID="$GPU_ID" bash generate/run_qwen_image_edit.sh "$@"
else
  echo "[2/4] Generate skipped (SKIP_GENERATE=true)"
fi

if [[ "$SKIP_INFER" != "true" ]]; then
  echo "[3/4] Closed-model inference (add_item only)..."
  MODELS="$MODELS" ONLY="add_item_figstep,add_item_mmsafety" bash scripts/run_meta_closed_infer.sh
else
  echo "[3/4] Infer skipped (SKIP_INFER=true)"
fi

if [[ "$SKIP_SCORE" != "true" ]]; then
  echo "[4/4] GPT judge scoring..."
  bash scripts/run_score_jsonl.sh
else
  echo "[4/4] Score skipped (SKIP_SCORE=true)"
fi

echo "=== Pipeline complete ==="
echo "Generations: outputs/generations/add_item_{figstep,mmsafety}/<model>.jsonl"
echo "Scores:      outputs/scores/<model>/*.summary.txt"
