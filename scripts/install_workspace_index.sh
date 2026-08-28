#!/usr/bin/env bash
# Install workspace index files next to this inference repo on Pasteur.
# Safe to re-run. Does not move data.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/artifacts"

if [[ -n "${MMBU_WORKSPACE_ROOT:-}" ]]; then
  WORKSPACE="$MMBU_WORKSPACE_ROOT"
elif [[ "$(basename "$REPO")" == "inference" ]]; then
  WORKSPACE="$(cd "$REPO/.." && pwd)"
else
  WORKSPACE=""
fi

if [[ -z "$WORKSPACE" || "$WORKSPACE" == "/" || ! -d "$WORKSPACE/src" ]]; then
  echo "sibling mmbu workspace not found; repo-local artifacts/ only ($REPO/artifacts)"
  WORKSPACE=""
fi

link_if_present() {
  local target="$1"
  local link="$2"
  if [[ -e "$target" || -d "$target" ]]; then
    ln -sfn "$target" "$link"
    echo "linked $link -> $target"
  else
    echo "skip symlink (missing $target)" >&2
  fi
}

if [[ -n "$WORKSPACE" ]]; then
  mkdir -p "$WORKSPACE/artifacts"
  ANALYSIS_CACHE="$WORKSPACE/src/analysis/finalized_analysis/llm_judge_cache"
  RESULTS="$WORKSPACE/results_cot_v3"
  link_if_present "$ANALYSIS_CACHE" "$REPO/artifacts/llm_judge_cache"
  link_if_present "$ANALYSIS_CACHE" "$WORKSPACE/artifacts/llm_judge_cache"
  link_if_present "$RESULTS" "$REPO/artifacts/results_cot_v3"
  link_if_present "$RESULTS" "$WORKSPACE/artifacts/results_cot_v3"

  cp -f "$REPO/CATALOG.md" "$WORKSPACE/CATALOG.md"
  cp -f "$REPO/docs/workspace-dropins/parent.code-workspace" "$WORKSPACE/mmbu.code-workspace"
  echo "copied CATALOG.md and mmbu.code-workspace to $WORKSPACE"

  if [[ -d "$WORKSPACE/src/analysis" ]]; then
    cp -f "$REPO/docs/workspace-dropins/analysis/AGENTS.md" "$WORKSPACE/src/analysis/AGENTS.md"
    mkdir -p "$WORKSPACE/src/analysis/src" "$WORKSPACE/src/analysis/finalized_analysis"
    cp -f "$REPO/docs/workspace-dropins/analysis/src/FROZEN.md" "$WORKSPACE/src/analysis/src/FROZEN.md"
    cp -f "$REPO/docs/workspace-dropins/analysis/finalized_analysis/LEGACY_JUDGE.md" \
      "$WORKSPACE/src/analysis/finalized_analysis/LEGACY_JUDGE.md"
    echo "installed analysis drop-ins"
  fi

  if [[ -d "$WORKSPACE/src/prepare_final_dataset" ]]; then
    cp -f "$REPO/docs/workspace-dropins/prepare_final_dataset/AGENTS.md" \
      "$WORKSPACE/src/prepare_final_dataset/AGENTS.md"
    echo "installed prepare_final_dataset AGENTS.md"
  fi
fi
