#!/usr/bin/env bash
# Run ON Pasteur as Ryan (gh auth as dcunhrya). Do not run from Cursor Cloud.
# 1) Tightens inference .gitignore and prepares a pasteur-sync branch.
# 2) Creates dcunhrya/mmbu-analysis (private) and pushes analysis code only.
set -euo pipefail

INFER="${INFER:-/pasteur/u/rdcunha/code/mmbu/inference}"
ANALYSIS="${ANALYSIS:-/pasteur/u/rdcunha/code/mmbu/src/analysis}"
NAME="${GIT_AUTHOR_NAME:-Ryan DCunha}"
EMAIL="${GIT_AUTHOR_EMAIL:-rdcunha@stanford.edu}"

die() { echo "error: $*" >&2; exit 1; }

command -v gh >/dev/null || die "gh CLI required (logged in as dcunhrya)"
gh auth status -h github.com | grep -q dcunhrya || {
  echo "gh is not dcunhrya; run: gh auth login" >&2
  gh auth status -h github.com || true
  exit 1
}

sync_inference() {
  cd "$INFER" || die "missing $INFER"
  git config user.name "$NAME"
  git config user.email "$EMAIL"
  git fetch origin
  git checkout -B pasteur-sync
  mkdir -p docs/workspace-dropins/analysis
  echo "Review git status; add code only (no results, venvs, caches)."
  git status -sb
  echo "Next: git add -A && git diff --cached --stat && git commit && git push -u origin pasteur-sync"
}

sync_analysis() {
  cd "$ANALYSIS" || die "missing $ANALYSIS"
  git config user.name "$NAME"
  git config user.email "$EMAIL"

  SRC_IGNORE="$INFER/docs/workspace-dropins/analysis/.gitignore"
  if [[ -f "$SRC_IGNORE" ]]; then
    cp -f "$SRC_IGNORE" "$ANALYSIS/.gitignore"
  fi

  if [[ ! -f "$ANALYSIS/.gitignore" ]]; then
    die "no analysis .gitignore"
  fi

  git add .gitignore
  git status -sb
  if ! git diff --cached --quiet; then
    git commit -m "Add gitignore: judge cache, DuckDB, venvs, eval-split outputs"
  fi

  if gh repo view dcunhrya/mmbu-analysis >/dev/null 2>&1; then
    echo "dcunhrya/mmbu-analysis already exists"
  else
    gh repo create dcunhrya/mmbu-analysis --private \
      --description "MMBU analysis: figures, eval-split, metrics (code only)" \
      --source=. --remote=origin --push
    echo "created and pushed https://github.com/dcunhrya/mmbu-analysis"
    return
  fi

  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin git@github.com:dcunhrya/mmbu-analysis.git
  git push -u origin HEAD
}

case "${1:-all}" in
  inference) sync_inference ;;
  analysis) sync_analysis ;;
  all)
    sync_inference
    echo "-----"
    sync_analysis
    ;;
  *) die "usage: $0 [all|inference|analysis]" ;;
esac
