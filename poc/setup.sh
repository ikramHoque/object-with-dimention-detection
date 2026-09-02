#!/usr/bin/env bash
# ============================================================================
# FILE PURPOSE · One-time environment setup. Run this before anything else.
#
# WHAT IT DOES
#   1. Installs uv (a fast Python package manager) if it is missing
#   2. Creates poc/.venv on Python 3.12
#   3. Installs everything in requirements.txt, including MoGe-2 from source
#   4. Registers a Jupyter kernel called "NX Survey POC"
#
# WHY IT EXISTS
#   The system Python on this machine is 3.9, which current torch will not run
#   on. Rather than upgrading the system Python, we build an isolated 3.12
#   environment so nothing else on the machine is affected.
#
# THIS IS BLOCKER #1 in STATUS.md — the pipeline cannot run until this succeeds.
#
# COST  ~20 minutes and ~2.5GB of disk, mostly torch.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

echo "==> creating .venv on python 3.12"
uv venv --python 3.12 .venv

echo "==> installing requirements (this pulls ~2.5GB of torch, be patient)"
uv pip install --python .venv/bin/python -r requirements.txt

echo "==> registering jupyter kernel"
.venv/bin/python -m ipykernel install --user --name nx-poc --display-name "NX Survey POC"

cat <<'MSG'

Setup complete.

  1. export ANTHROPIC_API_KEY=sk-ant-...      (needed for Arm A only)
  2. drop a room photo or video into poc/data/input/
  3. source poc/.venv/bin/activate && jupyter lab poc/pipeline.ipynb

Select the "NX Survey POC" kernel.
MSG
