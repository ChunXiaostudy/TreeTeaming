#!/bin/bash
# 批量生成 add_code 样本（数学公式 / 代码叠加，CPU-only）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "$SCRIPT_DIR/overlay_math_or_code.py" "$@"
