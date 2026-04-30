#!/bin/bash
# COMPLETE STARTER GUIDE - Nemotron-Indonesia 30B
# Run this on your 8x H200 machine

set -e

echo "=========================================="
echo "Nemotron-Indonesia Omni 30B-A3B - Setup & Run"
echo "=========================================="

# 1. CREATE PROJECT DIRECTORY
mkdir -p ~/nemotron-indonesia
cd ~/nemotron-indonesia

# 2. CREATE CONDA ENVIRONMENT
echo "Creating conda environment..."
conda create -n nemotron python=3.10 -y || true
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nemotron

# 3. INSTALL DEPENDENCIES
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2 datasets==3.2.0 accelerate==1.2.1
pip install peft==0.14.0 deepspeed==0.16.2 bitsandbytes==0.45.3
pip install sentencepiece protobuf scipy scikit-learn tqdm tensorboard

# Flash Attention 2 (optional but recommended for H200)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || echo "Flash Attention install failed, continuing without it"

# 4. VERIFY GPUs
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 5. DOWNLOAD BASE MODEL (30B)
echo ""
echo "Validating Nemotron-3-Nano-Omni-30B-A3B tokenizer..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('Tokenizer downloaded')
print('Model weights will download during first training/eval run; BF16 weights are ~62GB')
"

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review architecture: ARCHITECTURE.md"
echo "2. Check sources: python download_sources.py --sources core --dry-run"
echo "3. Pull first data: python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert"
echo "4. Prepare data: python prepare_data.py --datasets indo4b_hf wikipedia cc100 seapile"
echo "5. Start training: ./run_training.sh pretrain"
echo ""
echo "To activate environment later:"
echo "  conda activate nemotron"
echo ""
