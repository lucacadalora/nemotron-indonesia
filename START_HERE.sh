#!/bin/bash
# Nemotron-Indonesia setup checker.
# Run from the cloned repository: bash START_HERE.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"
MODEL_NAME="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

echo "=========================================="
echo "Nemotron-Indonesia Omni 30B-A3B - Setup Check"
echo "=========================================="
echo "Repo: $REPO_DIR"
echo ""

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not on PATH."
  echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Then rerun this script."
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU Status:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
else
  echo "WARNING: nvidia-smi not found. GPU driver/CUDA setup may be incomplete."
fi

echo ""
echo "Disk Status:"
df -h .

echo ""
echo "Creating/activating uv virtual environment: $VENV_DIR"
uv venv --python 3.10 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Detect system CUDA version and select the matching PyTorch wheel index.
# Pre-built stable wheels exist up to cu126 (PyTorch >=2.6).
# For cu130/cu131 we try the nightly index; if that also fails we abort with
# instructions rather than silently installing a mismatched binary.
# ---------------------------------------------------------------------------
echo ""
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || echo "unknown")
echo "Detected CUDA version: $CUDA_VERSION"

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

case "$CUDA_VERSION" in
  11.*) TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_PKG="torch==2.5.1" ;;
  12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_PKG="torch==2.5.1" ;;
  12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_PKG="torch" ;;
  12.5|12.6) TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        TORCH_PKG="torch" ;;
  12.*) TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        TORCH_PKG="torch"
        echo "NOTE: CUDA $CUDA_VERSION — no exact wheel; using cu126 (compatible within 12.x)." ;;
  13.*) TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu$(echo "$CUDA_VERSION" | tr -d '.')"
        TORCH_PKG="torch"
        echo "NOTE: CUDA $CUDA_VERSION — trying PyTorch nightly index (no stable wheel yet)." ;;
  unknown)
        echo "ERROR: Could not detect CUDA version (nvcc not found or output unparseable)."
        echo "       Install the CUDA toolkit and ensure nvcc is on PATH, then rerun."
        exit 1 ;;
  *)    echo "ERROR: Unrecognised CUDA version: $CUDA_VERSION"
        echo "       Add a case for this version in START_HERE.sh and rerun."
        exit 1 ;;
esac

echo "Installing PyTorch ($TORCH_PKG) for CUDA $CUDA_VERSION..."
if ! uv pip install "$TORCH_PKG" --index-url "$TORCH_INDEX"; then
  echo ""
  echo "ERROR: No pre-built PyTorch wheel found for CUDA $CUDA_VERSION."
  echo ""
  echo "Options:"
  echo "  1. Build PyTorch from source (4-8 hours, needs full CUDA $CUDA_VERSION toolkit):"
  echo "       git clone --recursive https://github.com/pytorch/pytorch"
  echo "       cd pytorch && pip install -r requirements.txt"
  echo "       USE_CUDA=1 TORCH_CUDA_ARCH_LIST=Auto python setup.py install"
  echo ""
  echo "  2. Install an older CUDA toolkit whose wheel is available, e.g. CUDA 12.6:"
  echo "       https://developer.nvidia.com/cuda-12-6-0-download-archive"
  echo "       Then re-run this script (the 13.x driver is backward-compatible)."
  exit 1
fi

echo ""
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# Build Flash Attention 2 from source so it links against the exact CUDA/PyTorch on this machine.
# This avoids pre-built wheel ABI mismatches. Compilation takes 10-30 minutes.
echo ""
echo "Installing Flash Attention 2 from source (this will take 10-30 minutes)..."
uv pip install ninja packaging wheel
MAX_JOBS=128 uv pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation \
  || echo "Flash Attention build failed; continuing without it."

echo ""
echo "Checking Python files..."
python -m py_compile download_sources.py prepare_data.py train_nemotron_indonesia.py evaluate.py

echo ""
echo "Validating Nemotron tokenizer..."
python - <<PY
from transformers import AutoTokenizer
model_name = "$MODEL_NAME"
print("loading tokenizer:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("tokenizer ok")
print("vocab size:", len(tokenizer))
PY

echo ""
echo "Checking first-milestone source links..."
python download_sources.py --sources first_milestone --verify-links --dry-run

echo ""
echo "=========================================="
echo "SETUP CHECK COMPLETE"
echo "=========================================="
echo ""
echo "Next commands:"
echo "1. Read workflow:           less OPERATOR_RUNBOOK.md"
echo "2. Pull first data:         python download_sources.py --sources first_milestone"
echo "3. Prepare first corpus:    python prepare_data.py --datasets indo4b_hf wikipedia"
echo "4. Run baseline eval:       python evaluate.py --model_path $MODEL_NAME --benchmark indonlu --output results/upstream_indonlu.json"
echo "5. Start smoke training:    ./run_training.sh pretrain"
echo ""
