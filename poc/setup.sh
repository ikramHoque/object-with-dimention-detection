#!/usr/bin/env bash
# One-time environment setup for the NX survey POC.
# System python here is 3.9 which is too old for current torch, so we pin 3.12 via uv.
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
