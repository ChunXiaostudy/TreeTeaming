#!/bin/bash
set -euo pipefail

# 需设置 OPENAI_API_KEY 与 OPENAI_BASE_URL（或 CLOSEDAPI_* 别名）
# 可选 JUDGE_MODEL（默认 gpt-4o-mini）
MODELS=${MODELS:-"gpt-4o-2024-05-13,claude-3-5-sonnet-20240620"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOVE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOVE_ROOT"

if [ -n "${ONLY:-}" ]; then
  python -u -m benchmark.meta_infer --models "$MODELS" --only "$ONLY"
else
  python -u -m benchmark.meta_infer --models "$MODELS"
fi

echo "Meta-attack closed inference done. JSONL at outputs/generations/{add_code,add_item}_{figstep,mmsafety}/<model>.jsonl"
