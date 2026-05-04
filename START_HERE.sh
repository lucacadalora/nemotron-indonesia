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

echo ""
echo "Installing Python dependencies..."
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# Flash Attention is useful but should not block the first workflow check.
echo ""
echo "Installing Flash Attention 2 if compatible..."
uv pip install flash-attn --no-build-isolation || echo "Flash Attention install failed; continuing without it."

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
