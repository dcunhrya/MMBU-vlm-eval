#!/usr/bin/env bash
# Run on Pasteur after ANTHROPIC_API_KEY is in inference/.env
set -euo pipefail

INFER="${INFER:-/pasteur/u/rdcunha/code/mmbu/inference}"
cd "$INFER"

export PYTHONPATH=src
export UV_CACHE_DIR="${UV_CACHE_DIR:-/pasteur/u/rdcunha/uv_cache}"
eval "$(python -c 'from mmbu.paths import apply_runtime_cache_env; apply_runtime_cache_env()')"

MODELS="gpt-5.6-sol,gpt-5.6-terra,claude-sonnet-5,claude-opus-5,Qwen3-VL-32B-Instruct,InternVL3_5-8B"

echo "=== v4 Sonnet rejudge status ==="
python scripts/run_sonnet_v4_open_rejudge.py \
  --judge-model claude-sonnet-5 \
  --model-filter "$MODELS" \
  --status-json | tee /tmp/v4_sonnet_judge_status.jsonl

echo "=== fill missing ==="
python scripts/run_sonnet_v4_open_rejudge.py \
  --judge-model claude-sonnet-5 \
  --model-filter "$MODELS"

echo "=== retry unparsed ==="
python scripts/retry_sonnet_unparsed_judge.py \
  --judge-model claude-sonnet-5 \
  --model-filter "$MODELS"

echo "=== PDF report ==="
python -m mmbu.v4_sonnet_report --rejoin

echo "Done. PDF at src/analysis/finalized_analysis/figures/split_v4_scores/public_private_scores_v4_sonnet_judge.pdf"
