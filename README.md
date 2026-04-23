# Nemotron-Indonesia 🇮🇩

**Indonesian sovereign LLM built on NVIDIA Nemotron — agentic AI for Bahasa Indonesia and local languages.**

Inspired by Malaysia's Ilmu-Nemo-30B, this project aims to build Indonesia's own agentic LLM using NVIDIA's Nemotron architecture.

## 🎯 Goals

1. **Base Model**: Fine-tune Nemotron on Indonesian corpus (20B+ tokens)
2. **Agentic Capabilities**: Tool use, reasoning, multi-step tasks
3. **Local Languages**: Indonesian, Javanese, Sundanese, Balinese, Minangkabau
4. **Benchmark**: Beat Sahabat AI on IndoMMLU
5. **Deployment**: API endpoint for Indonesian enterprises

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

### 3. Train

```bash
# Continued pre-training
./run_training.sh pretrain

# Supervised fine-tuning
./run_training.sh sft

# DPO alignment
./run_training.sh dpo
```

Or manually:

```bash
torchrun --nproc_per_node=8 train_nemotron_indonesia.py \
    --mode pretrain \
    --model_name nvidia/nemotron-3-8b-base-4k \
    --data_path ./data/processed \
    --output_dir ./models/nemotron-indonesia-8b \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --bf16 \
    --gradient_checkpointing
```

### 4. Evaluate

```bash
python evaluate.py \
    --model_path ./models/nemotron-indonesia-8b \
    --benchmark indommlu \
    --output results.json
```

## 📊 Training Configurations

### Continued Pre-training (8B → 8B-Indonesia)

| Parameter | Value |
|-----------|-------|
| Base model | Nemotron-3-8B-base-4k |
| Data | 20B tokens (OSCAR + CC100 + Wikipedia) |
| Batch size | 32 effective (4 per GPU × 8 GPUs) |
| Learning rate | 2e-5 |
| Epochs | 1 (20B tokens) |
| Time | ~2 days |
| Memory | ~60GB per GPU |

### Supervised Fine-tuning

| Parameter | Value |
|-----------|-------|
| Data | 500K instruction pairs |
| Batch size | 128 effective |
| Learning rate | 5e-6 |
| Epochs | 3 |
| Time | ~8 hours |

### DPO Alignment

| Parameter | Value |
|-----------|-------|
| Data | 50K preference pairs |
| Learning rate | 1e-6 |
| Epochs | 1 |
| Time | ~4 hours |

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
| GPT-4 (zero-shot) | ~55% |
| **Nemotron-Indonesia 8B** | **Target: 50%+** |

## 🤝 Comparison with Sahabat AI

| | Sahabat AI | Nemotron-Indonesia |
|---|---|---|
| **Base** | Llama 3 / Gemma | **Nemotron** |
| **Focus** | General chat | **Agentic AI** |
| **Size** | 8B, 70B | 8B (start), 30B+ (target) |
| **Local languages** | 5 | **10+** |
| **Commercial use** | Open source | **Open source** |
| **Inference** | Standard | **NVIDIA NIM optimized** |

## 📖 References

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
