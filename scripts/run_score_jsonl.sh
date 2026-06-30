#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOVE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MOVE_ROOT"

USE_MISSION="${USE_MISSION:-false}"
extra=()
if [[ "$USE_MISSION" == "true" ]]; then
  extra+=(--use-mission-objective)
fi

if [[ $# -gt 0 ]]; then
  python -u -m benchmark.score_jsonl --files "$@" "${extra[@]}"
else
  python -u -m benchmark.score_jsonl "${extra[@]}"
fi

echo "Scoring done. See outputs/scores/<model>/*.summary.txt"
