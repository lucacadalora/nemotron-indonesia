# Nemotron-Indonesia 🇮🇩

**Indonesian sovereign LLM built on NVIDIA Nemotron — agentic AI for Bahasa Indonesia and local languages.**

Inspired by Malaysia's Ilmu-Nemo-30B, this project aims to build Indonesia's own agentic LLM using NVIDIA's Nemotron architecture.

## 🎯 Goals

1. **Base model**: Fine-tune `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` (30B params, matching Ilmu-Nemo)
2. **Agentic capabilities**: Tool use, reasoning, multi-step tasks (like Ilmu-Nemo)
3. **Local languages**: Indonesian, Javanese, Sundanese, Balinese, Minangkabau, Buginese, Acehnese
4. **Benchmark**: Beat Sahabat AI 70B on IndoMMLU
5. **Deployment**: API endpoint + NIM microservices for Indonesian enterprises

## 🖥️ Hardware

- **8× NVIDIA H200** (141GB VRAM each)
- **Total VRAM**: 1,128 GB
- **Can handle**: Full fine-tuning of 30B model, or LoRA of 70B+

## 📁 Structure

```
nemotron-indonesia/
├── train_nemotron_indonesia.py    # Main training script
├── run_training.sh                # Quick start launcher
├── prepare_data.py               # Dataset curation
├── evaluate.py                   # IndoMMLU benchmark
├── data/                         # Training data
│   ├── raw/                      # Raw downloads
│   ├── processed/                # Tokenized data
│   └── instructions.jsonl        # Custom instruction data
├── models/                       # Output models
└── configs/                      # Training configs
    ├── pretrain.yaml
    ├── sft.yaml
    └── dpo.yaml
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n nemotron-indonesia python=3.10
conda activate nemotron-indonesia

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download and process Indonesian datasets
python prepare_data.py \
    --output_dir ./data/processed \
    --datasets oscar cc100 wikipedia kaskus liputan6 \
    --min_length 100 \
    --max_length 4096
```

### 3. Train (3 stages, ~5 days total on 8× H200)

```bash
# Stage 1: Continued pre-training (30B → 30B-Indonesia, ~3 days)
./run_training.sh pretrain

# Stage 2: Instruction fine-tuning (~12 hours)
./run_training.sh sft

# Stage 3: DPO alignment (~6 hours)
./run_training.sh dpo
```

Or manually:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_nemotron_indonesia.py \
    --mode pretrain \
    --model_name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --data_path ./data/processed \
    --output_dir ./models/nemotron-indonesia-30b \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.5e-5 \
    --num_epochs 1 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./configs/deepspeed_zero3.json
```

### 4. Evaluate

```bash
python evaluate.py \
    --model_path ./models/nemotron-indonesia-30b \
    --benchmark indommlu \
    --output results.json
```

## 📊 Training Configurations

### Continued Pre-training (30B → 30B-Indonesia)

| Parameter | Value |
|-----------|-------|
| Base model | **NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16** |
| Data | 20B tokens (OSCAR + CC100 + Wikipedia) |
| Batch size | 16 effective (2 per GPU × 8 GPUs) |
| Learning rate | 1.5e-5 |
| Epochs | 1 (20B tokens) |
| Time | **~3 days** |
| Memory | ~100GB per GPU (DeepSpeed ZeRO-3) |

### Supervised Fine-tuning

| Parameter | Value |
|-----------|-------|
| Data | 500K instruction pairs |
| Batch size | 32 effective (2 per GPU × 8 GPUs, grad accum 8) |
| Learning rate | 3e-6 |
| Epochs | 3 |
| Time | **~12 hours** |

### DPO Alignment

| Parameter | Value |
|-----------|-------|
| Data | 50K preference pairs |
| Learning rate | 5e-7 |
| Epochs | 1 |
| Time | **~6 hours** |

## 📚 Data Sources

### Pre-training
- [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) — 4B Indonesian tokens
- [CC100](https://huggingface.co/datasets/cc100) — Common Crawl Indonesian
- [Wikipedia ID](https://huggingface.co/datasets/wikimedia/wikipedia) — 500M tokens
- Kaskus forum corpus — Informal Indonesian
- [Liputan6](https://github.com/fajri91/Liputan6) — 215K news articles

### Fine-tuning
- [IndoMMLU](https://huggingface.co/datasets/indommlu) — 14K academic questions
- [NusaX](https://huggingface.co/datasets/indonlp/nusatranslation_senti) — 12 local languages
- [IndoNLI](https://huggingface.co/datasets/indonli) — Natural language inference
- Custom instruction data (government services, enterprise, healthcare)

## 🏆 Benchmarks

Target scores on IndoMMLU:

| Model | Accuracy |
|-------|----------|
| Sahabat AI 8B | ~45% |
| Sahabat AI 70B | ~52% |
| GPT-4 (zero-shot) | ~55% |
| **Nemotron-Indonesia 30B** | **Target: 55%+** |

## 🤝 Comparison with Sahabat AI & Ilmu-Nemo

| | Sahabat AI | Ilmu-Nemo (Malaysia) | **Nemotron-Indonesia** |
|---|---|---|---|
| **Base** | Llama 3 / Gemma | Nemotron 30B | **Nemotron 30B** |
| **Focus** | General chat | Agentic AI | **Agentic AI** |
| **Size** | 8B, 70B | **30B** | **30B** |
| **Languages** | 5 | Malay + English | **Indonesian + 10 local** |
| **Commercial use** | Open source | Open source | **Open source** |
| **Inference** | Standard | NVIDIA NIM | **NVIDIA NIM** |

## 💰 Your Cost

With **8× H200** you already own:
| Resource | Cloud Cost | Your Cost |
|----------|-----------|-----------|
| Pre-training (30B, 3 days) | ~$5,000 | **$0** (owned) |
| SFT (12 hours) | ~$1,000 | **$0** (owned) |
| DPO (6 hours) | ~$500 | **$0** (owned) |
| **Total** | **~$6,500** | **~$150 electricity** |

---

- [Ilmu-Nemo-30B (Malaysia)](https://theleaders-online.com/ytl-ai-labs-teams-up-with-nvidia-to-launch-ilmu%e2%80%91nemo%e2%80%9130b)
- [Sahabat AI (Indonesia)](https://sahabat-ai.com/)
- [NVIDIA Nemotron](https://developer.nvidia.com/nemotron)
- [Awesome Indonesian LLM Dataset](https://github.com/irfanfadhullah/awesome-indonesia-llm-dataset)

## 📝 License

Apache 2.0 — Open for commercial use.

## 🙏 Acknowledgments

- YTL AI Labs & NVIDIA for the Ilmu-Nemo inspiration
- Indosat & GoTo for Sahabat AI benchmark
- IndoNLP community for datasets
- Jatevo for compute infrastructure

---

**Built with ❤️ for Indonesia's AI sovereignty.**
