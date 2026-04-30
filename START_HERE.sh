#!/bin/bash
# Nemotron-Indonesia setup checker.
# Run from the cloned repository: bash START_HERE.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ENV_NAME="nemotron-indonesia"
MODEL_NAME="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

echo "=========================================="
echo "Nemotron-Indonesia Omni 30B-A3B - Setup Check"
echo "=========================================="
echo "Repo: $REPO_DIR"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not installed or not on PATH."
  echo "Install Miniconda/Mambaforge first, then rerun this script."
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
echo "Creating/activating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Flash Attention is useful but should not block the first workflow check.
echo ""
echo "Installing Flash Attention 2 if compatible..."
pip install flash-attn --no-build-isolation || echo "Flash Attention install failed; continuing without it."

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
