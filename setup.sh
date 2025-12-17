#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# Config (everything is relative to repo root)
# ==================================================
DEFAULT_ENV=".venv"
LLAVA_ENV="llava"

DEFAULT_LOCK="uv.lock"
DEFAULT_REQ="requirements-default.txt"
LLAVA_LOCK="uv-llava.lock"

LLAVA_REPO_URL="https://github.com/microsoft/LLaVA-Med.git"
LLAVA_DIR="LLaVA-Med"

# ==================================================
# Helpers
# ==================================================
info() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }
die()  { echo "[ERROR] $1"; exit 1; }

# ==================================================
# Preconditions
# ==================================================
command -v uv >/dev/null 2>&1 || die "uv is not installed or not on PATH"
command -v git >/dev/null 2>&1 || die "git is not installed"

[[ -f "$DEFAULT_LOCK" ]] || die "Missing $DEFAULT_LOCK"
[[ -f "$LLAVA_LOCK"   ]] || die "Missing $LLAVA_LOCK"

# ==================================================
# Create default uv environment (.venv)
# ==================================================
if [[ -d "$DEFAULT_ENV" ]]; then
    warn "Environment '$DEFAULT_ENV' already exists — skipping"
else
    info "Creating default uv environment ($DEFAULT_ENV)"
    uv venv --python 3.11 "$DEFAULT_ENV"
    source "$DEFAULT_ENV/bin/activate"
    uv pip install -r "$DEFAULT_REQ"
    deactivate
fi

# ==================================================
# Clone LLaVA-Med (repo root level)
# ==================================================
if [[ -d "$LLAVA_DIR" ]]; then
    warn "LLaVA-Med already exists — skipping clone"
else
    info "Cloning LLaVA-Med into repo root"
    git clone "$LLAVA_REPO_URL"
fi

# ==================================================
# Create LLaVA uv environment (llava/)
# ==================================================
if [[ -d "$LLAVA_ENV" ]]; then
    warn "Environment '$LLAVA_ENV' already exists — skipping"
else
    info "Creating LLaVA uv environment ($LLAVA_ENV) with Python 3.11"
    uv venv --python 3.11 "$LLAVA_ENV"
    source "$LLAVA_ENV/bin/activate"

    info "Installing LLaVA-Med from source"
    cd "$LLAVA_DIR"
    uv pip install .
    cd - >/dev/null

    info "Removing bitsandbytes (not needed / causes conflicts)"
    uv pip uninstall -y bitsandbytes

    deactivate
fi

info "Setup complete."