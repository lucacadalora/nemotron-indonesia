#!/bin/bash
# Quick start script for Nemotron-Indonesia training on 8× H200
# Usage: ./run_training.sh [pretrain|sft|dpo]

set -e

MODE=${1:-pretrain}
MODEL_NAME="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
DATA_PATH="./data"
OUTPUT_DIR="./models/nemotron-indonesia-omni-30b-${MODE}"

# H200 optimized settings
# 141GB VRAM per GPU, 8 GPUs = 1128GB total.
# Start conservatively for Omni; raise batch/context only after a memory smoke test.

BATCH_SIZE=2
GRAD_ACC=8
LR=1.5e-5
case "${MODE}" in
  pretrain) EPOCHS=1 ;;
  sft) EPOCHS=3 ;;
  dpo) EPOCHS=1 ;;
  *) echo "Usage: ./run_training.sh [pretrain|sft|dpo]"; exit 1 ;;
esac

# Check GPU availability
echo "🔍 Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${DATA_PATH}

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q torch transformers datasets accelerate peft bitsandbytes 2>/dev/null || true

# Run training with torchrun for distributed
echo "🚀 Starting ${MODE} adaptation on 8× H200 (Omni 30B-A3B model)..."
echo "Model: ${MODEL_NAME} (Omni 30B-A3B)"
echo "Output: ${OUTPUT_DIR}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACC * 8))"
echo "DeepSpeed: ZeRO-3"

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_id=nemotron-indonesia \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    train_nemotron_indonesia.py \
    --mode ${MODE} \
    --model_name ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --learning_rate ${LR} \
    --num_epochs ${EPOCHS} \
    --bf16 \
    --gradient_checkpointing \
    --flash_attention \
    --deepspeed ./configs/deepspeed_zero3.json

echo "✅ Training complete! Model saved to ${OUTPUT_DIR}"
echo "📊 Check results.txt for IndoMMLU benchmark scores"
