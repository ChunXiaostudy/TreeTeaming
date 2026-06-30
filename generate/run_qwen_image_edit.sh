#!/bin/bash
# 批量生成 add_item 样本（Qwen-Image-Edit，本地离线）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOVE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
META_ROOT="$MOVE_ROOT/Meta-attack"

MODEL_DIR="${MODEL_DIR:-${QWEN_IMAGE_EDIT_PATH:-Qwen/Qwen-Image-Edit}}"
FIGSTEP_DIR="${FIGSTEP_DIR:-$META_ROOT/base/figstep}"
FIGSTEP_OUT="${FIGSTEP_OUT:-$META_ROOT/add_item/figstep}"
MMSAFETY_DIR="${MMSAFETY_DIR:-$META_ROOT/base/mmsafety}"
MMSAFETY_OUT="${MMSAFETY_OUT:-$META_ROOT/add_item/mmsafety}"

GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bf16}"
NUM_STEPS="${NUM_STEPS:-28}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
SEED="${SEED:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-deformed, blurry, low quality, extra objects, cropped}"
OVERWRITE="${OVERWRITE:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --figstep-dir) FIGSTEP_DIR="$2"; shift 2 ;;
    --figstep-out) FIGSTEP_OUT="$2"; shift 2 ;;
    --mmsafety-dir) MMSAFETY_DIR="$2"; shift 2 ;;
    --mmsafety-out) MMSAFETY_OUT="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --num-steps) NUM_STEPS="$2"; shift 2 ;;
    --true-cfg-scale) TRUE_CFG_SCALE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --negative-prompt) NEGATIVE_PROMPT="$2"; shift 2 ;;
    --overwrite) OVERWRITE="true"; shift 1 ;;
    -h|--help)
      echo "用法: $0 [--model-dir PATH] [--figstep-dir PATH] [--figstep-out PATH] [--mmsafety-dir PATH] [--mmsafety-out PATH] [--gpu N] [--dtype bf16|fp16|fp32] [--num-steps N] [--true-cfg-scale F] [--seed N] [--negative-prompt TEXT] [--overwrite]"
      exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy || true

mkdir -p "$FIGSTEP_OUT" "$MMSAFETY_OUT"

cmd=(
  python -u "$SCRIPT_DIR/batch_qwen_image_edit.py"
  --model-dir "$MODEL_DIR"
  --figstep-dir "$FIGSTEP_DIR"
  --figstep-out "$FIGSTEP_OUT"
  --mmsafety-dir "$MMSAFETY_DIR"
  --mmsafety-out "$MMSAFETY_OUT"
  --gpu "$GPU_ID"
  --dtype "$DTYPE"
  --num-steps "$NUM_STEPS"
  --true-cfg-scale "$TRUE_CFG_SCALE"
  --seed "$SEED"
  --negative-prompt "$NEGATIVE_PROMPT"
  --xformers
)

if [[ "$OVERWRITE" == "true" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
