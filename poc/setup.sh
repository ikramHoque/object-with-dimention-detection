#!/usr/bin/env bash
# ============================================================================
# FILE PURPOSE · One-time environment setup. Run this before anything else.
#
# WHAT IT DOES
#   1. Installs uv (a fast Python package manager) if it is missing
#   2. Creates poc/.venv on Python 3.12
#   3. Installs requirements.txt
#   4. Installs MoGe-2 separately, pinned and --no-deps (requirements.txt
#      explains why this cannot be folded into step 3)
#   5. Registers a Jupyter kernel called "NX Survey POC"
#   6. Verifies every import works, so a broken env fails loudly and early
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

# Idempotent on purpose: the first thing anyone does after a failed setup is run
# it again, and `uv venv` hard-fails on an existing directory. Reuse a .venv that
# is already on the right Python, recreate it if not, and allow a forced rebuild
# with NX_RECREATE_VENV=1.
VENV_PY="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
if [ "${NX_RECREATE_VENV:-0}" = "1" ]; then
  echo "==> recreating .venv (NX_RECREATE_VENV=1)"
  uv venv --clear --python 3.12 .venv
elif [ -z "$VENV_PY" ]; then
  echo "==> creating .venv on python 3.12"
  uv venv --python 3.12 .venv
elif [ "$VENV_PY" != "3.12" ]; then
  echo "==> replacing .venv (found python $VENV_PY, need 3.12)"
  uv venv --clear --python 3.12 .venv
else
  echo "==> reusing existing .venv on python $VENV_PY"
fi

# ---------------------------------------------------------------------------
# Platform detection.
#
# WHAT VARIES BY PLATFORM: only which torch *wheel* we fetch.
# WHAT DOES NOT VARY: the models themselves. Every platform runs the same
#   MoGe-2 and the same detectors, deliberately. See the POLICY note at the
#   bottom of this file before you change that.
#
# Written for bash 3.2, which is what macOS still ships, so no fancy arrays.
# TORCH_ARGS is intentionally expanded unquoted below: it holds fixed literal
# flags with no spaces, and must vanish entirely when empty.
# ---------------------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
TORCH_ARGS=""

case "$OS" in
  Darwin)
    # The macOS arm64 wheel is the only torch build Apple silicon has, and MPS
    # is already compiled into it. Passing --torch-backend here would at best
    # do nothing and at worst select a CPU-only index, so we pass nothing.
    ACCEL="mps — Apple GPU, already in the default wheel"
    ;;
  Linux|MINGW*|MSYS*|CYGWIN*)
    if command -v nvidia-smi >/dev/null 2>&1; then
      # `auto` asks uv to read the installed driver and pick the matching CUDA
      # wheel (cu126/cu128/...) rather than us hard-coding a version that goes
      # stale. This is the same class of mistake that MoGe-3's cu130 pin makes.
      ACCEL="cuda — nvidia-smi present, letting uv match the driver"
      TORCH_ARGS="--torch-backend auto"
    else
      # Explicitly CPU: on Linux the default PyPI torch bundles CUDA and is
      # several GB larger for no benefit without a GPU.
      ACCEL="cpu — no nvidia-smi found"
      TORCH_ARGS="--torch-backend cpu"
    fi
    ;;
  *)
    ACCEL="unknown platform — falling back to the default wheel"
    ;;
esac

echo "==> platform: $OS/$ARCH  ->  $ACCEL"

echo "==> installing requirements (this pulls ~2.5GB of torch, be patient)"
# shellcheck disable=SC2086  # unquoted on purpose, see TORCH_ARGS note above
uv pip install --python .venv/bin/python $TORCH_ARGS -r requirements.txt

# Pinned to MoGe v2.0.0 and --no-deps ON PURPOSE. MoGe's main branch is MoGe-3,
# which forces a CUDA-only torch wheel and cannot resolve on macOS at all.
# The full explanation is in requirements.txt - read it before changing this.
MOGE_PIN=b942f00bdc2a2a23ebb474fbe034d487e6dcceec
echo "==> installing MoGe-2 (pinned ${MOGE_PIN:0:7}, --no-deps)"
uv pip install --python .venv/bin/python --no-deps \
  "git+https://github.com/microsoft/MoGe.git@$MOGE_PIN"

echo "==> registering jupyter kernel"
.venv/bin/python -m ipykernel install --user --name nx-poc --display-name "NX Survey POC"

echo "==> verifying the environment"
.venv/bin/python - <<'VERIFY'
import importlib, sys
# Import-only check: no weights are downloaded here. The point is that a broken
# environment fails now, in two seconds, not 20 minutes into a real run.
mods = ["numpy", "cv2", "torch", "torchvision", "transformers",
        "huggingface_hub", "timm", "scipy", "utils3d", "anthropic",
        "moge.model.v2"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok    {m}")
    except Exception as e:
        bad.append(m)
        print(f"  FAIL  {m}: {type(e).__name__}: {e}")
import torch
dev = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"  torch {torch.__version__} on {dev}")
if dev == "cpu":
    print("  ! no GPU backend - expect ~15s per image instead of ~3s")
if bad:
    print()
    print(f"  {len(bad)} import(s) failed: {', '.join(bad)}")
    print("  See the troubleshooting table in RUNNING.md.")
    sys.exit(1)
VERIFY

cat <<'MSG'

Setup complete.

  1. export ANTHROPIC_API_KEY=sk-ant-...      (needed for Method A only)
  2. drop a room photo or video into poc/data/input/
  3. cp a room photo into poc/pipelines/grounding_dino__moge2/data/input/
  4. python -m poc.pipelines.grounding_dino__moge2.run          (one shot)
     or jupyter lab poc/pipelines/grounding_dino__moge2/notebook.ipynb

Select the "NX Survey POC" kernel.
MSG

# ============================================================================
# POLICY · why this script does NOT install a different model per platform
#
# It is tempting to have Linux/CUDA machines use MoGe-3 (the current MoGe main
# branch) and Macs fall back to MoGe-2. Do not do that.
#
# MoGe-2 and MoGe-3 are DIFFERENT MODELS with different accuracy. This project's
# entire deliverable is an accuracy number — bias, MAPE, P90, capacity breach.
# If the Mac measures with MoGe-2 and Colab measures with MoGe-3, those numbers
# are not comparable, and no one can tell whether a change came from the code,
# the model, or which laptop happened to run it. Silent per-machine model
# substitution is the fastest way to make a whole R&D result meaningless.
#
# So: same models everywhere, and the platform only decides which torch wheel
# to download.
#
# If we DO want MoGe-3, it gets added as its own registry entry (`moge3`) and
# its own combination row, chosen explicitly with `--depth moge3`, reported
# separately, and marked unavailable on platforms that cannot run it. That is
# already how `unidepth2` is handled for licence reasons — same pattern, a
# different reason.
#
# For the record, MoGe-3 needs more than "not a Mac": flex-gemm compiles CUDA
# kernels with nvcc at install time and needs Triton, which has no macOS build.
# So the real dividing line is an NVIDIA GPU with a CUDA toolchain, not the OS.
# Colab T4 qualifies; a CPU-only Linux box does not.
# ============================================================================
