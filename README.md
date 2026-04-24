# Nemotron-Indonesia

**Indonesian sovereign LLM built on NVIDIA Nemotron — agentic AI for Bahasa Indonesia and 10+ local languages.**

Inspired by Malaysia's Ilmu-Nemo-30B (YTL AI Labs x NVIDIA, March 2026), this project builds Indonesia's own agentic LLM using NVIDIA's Nemotron architecture. Agentic capabilities include tool use, multi-step reasoning, and function calling — beyond basic chat.

---

## Goals

1. **Base model**: Continue pre-training `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` on rigorous Indonesian corpus
2. **Agentic capabilities**: Tool use, reasoning, multi-step task execution
3. **Local languages**: Indonesian, Javanese, Sundanese, Balinese, Minangkabau, Buginese, Acehnese, Banjarese, Ngaju Dayak, Madurese
4. **Benchmark target**: 55%+ on IndoMMLU (beat Sahabat AI 70B at ~52%, match SEA-LION v3-9B at ~55%)
5. **Deployment**: API endpoint + NIM microservices for Indonesian enterprises

---

## Hardware

- **8x NVIDIA H200** (141GB VRAM each, 1,128 GB total)
- **DeepSpeed ZeRO-3** for distributed training
- **NVLink + NVSwitch** interconnect
- Estimated training time: ~5 days (3 days pre-train + 12h SFT + 6h DPO)

---

## Repository Structure

```
nemotron-indonesia/
|-- [PRD.md](PRD.md)                          # Full requirements document
|-- [DATA_STRATEGY.md](DATA_STRATEGY.md)                # Rigorous data curation strategy
|-- [train_nemotron_indonesia.py](train_nemotron_indonesia.py)     # Main training pipeline (3 stages)
|-- [run_training.sh](run_training.sh)                 # Quick start launcher
|-- [prepare_data.py](prepare_data.py)                 # Dataset curation + quality pipeline
|-- [multica_sync.py](multica_sync.py)           # 🆕 Sync progress to Multica PM
|-- [evaluate.py](evaluate.py)                     # IndoMMLU + SEA-HELM benchmarks
|-- [requirements.txt](requirements.txt)                # Python dependencies
|-- configs/
|   |-- deepspeed_zero3.json        # DeepSpeed ZeRO-3 config
|   |-- deepspeed_zero3_gpuonly.json
|-- data/                           # Training data (generated)
|-- models/                         # Output checkpoints (generated)
```

---

## Track Progress in Multica

You can track Nemotron pipeline progress in your [Multica](https://multica.jatevo.ai) instance:

```bash
# 1. Setup: Create project + tasks in Multica
python multica_sync.py --setup \
    --api-token YOUR_MULTICA_TOKEN \
    --workspace-id YOUR_WORKSPACE_ID

# 2. Update status when a phase completes
python multica_sync.py --update-phase data_prep --status done
python multica_sync.py --update-phase pretrain --status in_progress

# 3. View dashboard
python multica_sync.py --dashboard
```

This creates 5 pipeline phases as tasks in Multica and syncs status as you progress.

**Get your API token:** Multica UI → Settings → API Tokens  
**Get workspace ID:** From URL `/workspace/WORKSPACE_ID`

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n nemotron-indonesia python=3.10
conda activate nemotron-indonesia
pip install -r requirements.txt
```

Alternatively, we can use `uv` to prepare the environment.

```bash
uv venv nemotron-indonesia --python 3.10
source nemotron-indonesia/bin/activate
uv pip install -r requirements.txt
```

Error fix when installing `flash-attn` as follows.
```bash
sudo apt install python3-dev
uv pip install torch psutil numpy
uv pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Data Preparation

```bash
# Full pipeline with NER quality filter (recommended for production)
python prepare_data.py \
    --output_dir ./data/processed \
    --datasets oscar cc100 wikipedia \
    --use_ner_filter \
    --quality_threshold 0.1

# Quick mode (skip NER, for initial exploration)
python prepare_data.py \
    --output_dir ./data/processed \
    --datasets wikipedia liputan6
```

**Pipeline phases (all on your server):**

| Phase | What happens | Time |
|-------|-------------|------|
| **1. DOWNLOAD** | Fetch from HuggingFace (OSCAR, CC100, Wikipedia, etc.) | ~10-30 min |
| **2. CLEAN** | Regex filters, length checks, language detection | ~5 min |
| **3. QUALITY** | Optional: NER entity-density scoring (BERT-based) | ~20 min |
| **4. PACKAGE** | Deduplication, tokenization, language tags, save | ~10 min |

**The NER filter** (`--use_ner_filter`) uses `cahya/bert-base-indonesian-NER` to score
documents by entity density. Documents rich in people, organizations, and locations
(like news, Wikipedia) score higher. Entity-sparse documents (spam, noise) are filtered out.

See [DATA_STRATEGY.md](DATA_STRATEGY.md) for the full rigorous pipeline including:
- Multi-signal quality scoring (perplexity, toxicity, readability, **NER entity density**)
- 3-level deduplication (exact + MinHash LSH + fuzzy simhash)
- Benchmark decontamination (13-gram Bloom filter against IndoMMLU/NusaX test sets)
- Curriculum learning (easy to medium to hard)

### 3. Training (3 Stages)

```bash
# Stage 1: Continued pre-training (~3 days)
./run_training.sh pretrain

# Stage 2: Supervised fine-tuning (~12 hours)
./run_training.sh sft

# Stage 3: DPO alignment (~6 hours)
./run_training.sh dpo
```

Or manually with DeepSpeed:

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

### 4. Evaluation

```bash
# IndoMMLU (primary benchmark)
python evaluate.py \
    --model_path ./models/nemotron-indonesia-30b \
    --benchmark indommlu \
    --output results_indommlu.json

# SEA-HELM / BHASA (SEA multilingual benchmark)
python evaluate.py \
    --model_path ./models/nemotron-indonesia-30b \
    --benchmark sea-helm \
    --output results_seahelm.json
```

---

## Training Configurations

### Stage 1: Continued Pre-training (30B → 30B-Indonesia)

| Parameter | Value |
|-----------|-------|
| Base model | NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 |
| Data | 20B tokens (rigorous mix, see DATA_STRATEGY.md) |
| Batch size | 128 effective (2 per GPU x 8 GPUs x grad accum 8) |
| Learning rate | 1.5e-5 (cosine schedule) |
| Epochs | 1 |
| Time | ~3 days |
| Memory per GPU | ~100GB (DeepSpeed ZeRO-3) |

### Stage 2: Supervised Fine-tuning

| Parameter | Value |
|-----------|-------|
| Data | 500K instruction pairs (IndoMMLU train + synthetic textbooks + custom) |
| Batch size | 128 effective |
| Learning rate | 3e-6 |
| Epochs | 3 |
| Time | ~12 hours |

### Stage 3: DPO Alignment

| Parameter | Value |
|-----------|-------|
| Data | 50K preference pairs |
| Learning rate | 5e-7 |
| Beta | 0.1 |
| Epochs | 1 |
| Time | ~6 hours |

---

## Rigorous Data Strategy

See [DATA_STRATEGY.md](DATA_STRATEGY.md) for the complete strategy. Key differentiators from Sahabat AI:

| Rigorous Step | Sahabat AI | Nemotron-Indonesia |
|--------------|-----------|-------------------|
| Benchmark decontamination | Unknown | 13-gram Bloom filter removal |
| Quality scoring | Basic | 6-signal composite score |
| Deduplication | Single-pass | 3-level (exact + LSH + fuzzy) |
| Educational content | ~2% | 25%+ (BSE textbooks, academic papers) |
| Curriculum learning | No | Easy to medium to hard sorting |
| Synthetic data | None | 100K textbook Q&A from BSE |

### Pre-training Corpus (20B Tokens)

| Source | Tokens | % | Quality | Purpose |
|--------|--------|---|---------|---------|
| Indonesian Wikipedia | 2.0B | 10% | A+ | Structured knowledge |
| Indonesian Academic Corpus | 2.5B | 12.5% | A+ | STEM, textbooks, papers |
| OSCAR (filtered) | 5.0B | 25% | B+ | Broad web coverage |
| CC100 Indonesian (filtered) | 3.0B | 15% | B+ | Web diversity |
| SEA-LION Pile - Indonesian | 1.0B | 5% | A | AI Singapore curated |
| Liputan6 + ID News | 2.0B | 10% | A | Formal news, social science |
| Kaskus (heavily filtered) | 1.0B | 5% | C+ | Informal Indonesian |
| Indonesian Government Docs | 1.5B | 7.5% | A | Legal, civics, policy |
| Religious Texts (ID) | 0.5B | 2.5% | A | Cultural literacy |
| Javanese/Sundanese/Balinese | 1.5B | 7.5% | B | Local language coverage |
| English Academic (STEM) | 1.0B | 5% | A+ | STEM concept transfer |
| **TOTAL** | **20.0B** | **100%** | | |

### Fine-tuning Data (500K Pairs)

| Source | Pairs | Type |
|--------|-------|------|
| IndoMMLU train split | 50K | Academic Q&A |
| Synthetic BSE textbooks | 100K | Curriculum-aligned Q&A |
| NusaX sentiment + NLI | 30K | Classification |
| Translated Flan/CoT | 150K | Reasoning tasks |
| Government FAQ | 50K | Civic, legal, admin |
| Custom agentic instructions | 120K | Tool use, multi-step |

---

## Benchmarks

### Primary: IndoMMLU

| Model | Size | Accuracy |
|-------|------|----------|
| Sahabat AI 8B | 8B | ~45% |
| Sahabat AI 70B | 70B | ~52% |
| SEA-LION v3-9B | 9B | ~55% |
| GPT-4 (zero-shot) | - | ~55% |
| **Nemotron-Indonesia 30B** | **30B** | **Target: 55%+** |

### Secondary

- **SEA-HELM / BHASA**: SEA multilingual benchmark (target: competitive with SEA-LION)
- **NusaX Sentiment**: 12 local languages (target: 80%+)
- **IndoNLI**: Natural language inference (target: 75%+)
- **IndoSum**: Summarization (target: ROUGE-L 35+)
- **Custom Agentic Eval**: Tool use, multi-step reasoning

---

## Competitive Analysis

| Attribute | Sahabat AI | SEA-LION | Ilmu-Nemo | Nemotron-Indonesia |
|-----------|-----------|----------|-----------|-------------------|
| Base Model | Llama 3 / Gemma | MPT/Llama/Gemma | Nemotron 30B | **Nemotron 30B** |
| Size | 8B, 70B | 3B-9B | 30B | **30B** |
| Focus | General chat | SEA multilingual | Agentic AI | **Agentic AI** |
| Indonesia-Specific | Yes | Partial | No | **Yes** |
| Local Languages | 5 | 11 | Malay + EN | **10+** |
| Agentic Tools | No | No | Yes | **Yes** |
| Gov Backing | Private (GoTo) | Singapore Gov (NRF) | Private (YTL) | **Jatevo + partners** |
| Commercial License | Open source | MIT/Open | Open source | **Open source** |
| Inference Stack | Standard | Standard | NVIDIA NIM | **NVIDIA NIM** |

### Collaboration Opportunities

- **AI Singapore**: Proven collaborator (co-built Sahabat AI v1/v2). SEA-LION Pile dataset is open source. Potential three-way partnership: NVIDIA + AI Singapore + Jatevo.
- **SEA-LION Leaderboard**: Submit Nemotron-Indonesia for official ranking alongside SEA-LION models.
- **IndoNLP Community**: Leverage NusaX, IndoNLI, IndoSum benchmarks and datasets.

---

## Cost Analysis (Owned Hardware)

| Stage | Cloud Equivalent | Your Cost (Electricity) |
|-------|-----------------|------------------------|
| Data preparation | ~$500 | ~$10 |
| Pre-training (3 days) | ~$5,000 | ~$30 |
| SFT (12 hours) | ~$1,000 | ~$5 |
| DPO (6 hours) | ~$500 | ~$3 |
| **Total** | **~$7,000** | **~$50** |

---

## References

- [Ilmu-Nemo-30B (Malaysia)](https://theleaders-online.com/ytl-ai-labs-teams-up-with-nvidia-to-launch-ilmu%e2%80%91nemo%e2%80%9130b)
- [Sahabat AI (Indonesia)](https://sahabat-ai.com/)
- [SEA-LION (AI Singapore)](https://huggingface.co/collections/aisingapore/sea-lionv3-672589a39cdadd6a5b199581)
- [NVIDIA Nemotron](https://developer.nvidia.com/nemotron)
- [Awesome Indonesian LLM Dataset](https://github.com/irfanfadhullah/awesome-indonesia-llm-dataset)
- [IndoMMLU Paper](https://arxiv.org/abs/2310.04928)

---

## License

Apache 2.0 — Open for commercial use.

## Acknowledgments

- YTL AI Labs and NVIDIA for the Ilmu-Nemo inspiration
- Indosat and GoTo for Sahabat AI benchmark
- AI Singapore for SEA-LION Pile dataset and BHASA benchmark
- IndoNLP community for datasets and evaluation frameworks
- Jatevo for compute infrastructure

---

Built for Indonesia's AI sovereignty.
