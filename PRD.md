# Nemotron-Indonesia 30B — Product Requirements Document

**Version:** 1.0  
**Date:** April 23, 2026  
**Status:** Pipeline Ready  
**Owner:** Luca Cada Lora (Jatevo)  
**Architecture Page:** https://diagrams.jatevo.ai/nemotron-indonesia.html

---

## 1. Executive Summary

Nemotron-Indonesia is a sovereign Indonesian large language model built on NVIDIA's Nemotron architecture. Inspired by Malaysia's Ilmu-Nemo-30B (YTL AI Labs × NVIDIA, launched March 2026), this project aims to create Indonesia's first agentic LLM — capable of tool use, multi-step reasoning, and enterprise automation — while deeply understanding Bahasa Indonesia and 10+ local languages.

**Key differentiator from Sahabat AI:** Sahabat AI (Indosat + GoTo) is built on Llama/Gemma and focuses on general chat. Nemotron-Indonesia uses NVIDIA's Nemotron architecture, which is optimized for agentic AI — tool use, reasoning, and function calling.

---

## 2. Goals & Objectives

### Primary Goals
1. **Build a 30B parameter Indonesian LLM** that surpasses Sahabat AI 70B on IndoMMLU benchmark
2. **Enable agentic capabilities** — tool use, reasoning, multi-step task execution
3. **Support 10+ Indonesian local languages** beyond just standard Indonesian
4. **Deploy as sovereign infrastructure** — data stays in Indonesia, no foreign API dependency
5. **Open source** the model weights and training recipe for community use

### Success Metrics
| Metric | Target | Benchmark |
|--------|--------|-----------|
| IndoMMLU Accuracy | 55%+ | Sahabat AI 70B: 52% |
| NusaX Multilingual | 80%+ | Multi-language sentiment |
| Inference Latency | <100ms/token | H200 optimized |
| Model Size | 30B params | Match Ilmu-Nemo |
| Context Window | 4,096 tokens | Base model spec |

---

## 3. Competitive Analysis

### 3.1 Sahabat AI (Existing Indonesian LLM)
- **Builder:** Indosat Ooredoo Hutchison + GoTo
- **Base:** Llama 3 (8B, 70B) + Gemma (9B)
- **Data:** 50B tokens pre-training + 448K instruction pairs
- **Focus:** General chat, multilingual conversation
- **Strengths:** First-mover, big tech backing, established ecosystem
- **Weaknesses:** Not agentic, Llama commercial license restrictions, limited local languages

### 3.2 SEA-LION (ASEAN Regional Benchmark)
- **Builder:** AI Singapore (NUS-hosted, NRF-funded)
- **Base:** MPT (3B, 7B) → Llama 3 (8B) → Gemma 2 (9B)
- **Data:** 980B tokens (v1) / 50B+ tokens (v3) across 11 SEA languages
- **Focus:** Southeast Asian multilingual LLM
- **Strengths:** Government-backed, open source, SEA-LION Pile dataset, proven collaborator (co-built Sahabat AI)
- **Weaknesses:** Not Indonesia-specific, smaller model sizes, not agentic
- **Collaboration potential:** AI Singapore is a natural partner — they already collaborated with GoTo on Sahabat AI

### 3.3 Ilmu-Nemo-30B (Malaysian Benchmark)
- **Builder:** YTL AI Labs × NVIDIA
- **Base:** NVIDIA Nemotron 30B
- **Launch:** March 17, 2026
- **Focus:** Agentic AI for Malaysia
- **Achievement:** +23% on MalayMMLU vs base model
- **Infrastructure:** YTL AI Cloud (NVIDIA DGX)

### 3.4 Nemotron-Indonesia Positioning
| Attribute | Sahabat AI | SEA-LION | Ilmu-Nemo | Nemotron-Indonesia |
|-----------|-----------|----------|-----------|-------------------|
| **Base Model** | Llama 3 / Gemma | MPT/Llama/Gemma | Nemotron 30B | **Nemotron 30B** |
| **Size** | 8B, 70B | 3B, 7B, 8B, 9B | **30B** | **30B** |
| **Focus** | General chat | SEA multilingual | Agentic AI | **Agentic AI** |
| **Local Languages** | 5 | 11 | Malay + EN | **10+** |
| **Indonesia-Specific** | Yes | Partial | No | **Yes** |
| **Commercial License** | Open source | MIT/Open | Open source | **Open source** |
| **Inference Stack** | Standard | Standard | NVIDIA NIM | **NVIDIA NIM** |
| **Agentic Tools** | No | No | Yes | **Yes** |
| **Gov Backing** | Private (GoTo) | Singapore Gov | Private (YTL) | **Jatevo + partners** |

---

## 4. Technical Architecture

### 4.1 Hardware Infrastructure
```
Cluster Configuration:
├── 8× NVIDIA H200 (141GB VRAM each)
├── Total VRAM: 1,128 GB
├── Interconnect: NVLink + NVSwitch
├── Network: InfiniBand NDR
└── Storage: NVMe SSD for dataset caching
```

### 4.2 Memory Requirements for 30B Model
Without DeepSpeed ZeRO-3:
- Model weights (BF16): 60GB
- Gradients (BF16): 60GB
- Optimizer states (FP32): 120GB
- Activations: ~70GB
- **Total per GPU: ~240GB** ❌ Exceeds H200 141GB

With DeepSpeed ZeRO-3:
- Weights partitioned: 60GB ÷ 8 = 7.5GB/GPU
- Gradients partitioned: 60GB ÷ 8 = 7.5GB/GPU
- Optimizer partitioned: 120GB ÷ 8 = 15GB/GPU
- Activations: ~70GB/GPU
- **Total per GPU: ~100GB** ✅ Fits in 141GB

### 4.3 Training Pipeline (3 Stages)

#### Stage 1: Continued Pre-training
- **Input:** Nemotron-3-Nano-30B-A3B-Base-BF16
- **Data:** 20B Indonesian tokens (OSCAR + CC100 + Wikipedia + Kaskus + Liputan6)
- **Config:**
  - Batch size: 2 per GPU × 8 GPUs × grad accum 8 = effective 128
  - Learning rate: 1.5e-5 (cosine schedule)
  - Epochs: 1 (single pass over 20B tokens)
  - Precision: BF16
  - DeepSpeed: ZeRO-3
- **Output:** Nemotron-Indonesia-30B-Base
- **Time:** ~3 days

#### Stage 2: Supervised Fine-tuning (SFT)
- **Input:** Nemotron-Indonesia-30B-Base
- **Data:** 500K instruction pairs (IndoMMLU + NusaX + custom instructions)
- **Config:**
  - Batch size: 2 per GPU × 8 GPUs × grad accum 8 = effective 128
  - Learning rate: 3e-6 (lower than pre-train)
  - Epochs: 3
  - Chat template with special tokens: `<|user|>`, `<|assistant|>`, `<|system|>`
  - Language tokens: `<|id|>`, `<|jv|>`, `<|su|>`, etc.
- **Output:** Nemotron-Indonesia-30B-SFT
- **Time:** ~12 hours

#### Stage 3: Direct Preference Optimization (DPO)
- **Input:** Nemotron-Indonesia-30B-SFT
- **Data:** 50K preference pairs (chosen vs rejected responses)
- **Config:**
  - Batch size: 2 per GPU × 8 GPUs × grad accum 8 = effective 128
  - Learning rate: 5e-7 (very low for stability)
  - Beta: 0.1 (DPO temperature)
  - Epochs: 1
- **Output:** Nemotron-Indonesia-30B-DPO (final model)
- **Time:** ~6 hours

### 4.4 Software Stack
| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.5.1 | Deep learning framework |
| Transformers | 4.46.2 | Model loading and training |
| DeepSpeed | 0.16.2 | Distributed training (ZeRO-3) |
| Flash Attention | 2.4.0 | Memory-efficient attention |
| Datasets | 3.2.0 | Data loading and processing |
| Accelerate | 1.2.1 | HuggingFace distributed |
| PEFT | 0.14.0 | LoRA (if needed for experiments) |

---

## 5. Dataset Strategy

### 5.1 Pre-training Corpus (20B+ tokens)
| Source | Size | Type | Quality |
|--------|------|------|---------|
| **OSCAR** | 4B tokens | Web crawl (Indonesian) | Medium |
| **CC100** | 2B tokens | Common Crawl Indonesian | Medium |
| **Wikipedia ID** | 500M tokens | Encyclopedia | High |
| **Kaskus** | Large | Forum/Informal Indonesian | Low-Medium |
| **Liputan6** | 215K articles | News corpus | High |

**Processing Pipeline:**
1. Download raw datasets
2. Clean (remove URLs, emails, excessive whitespace)
3. Filter (min length 100 chars, Indonesian language detection)
4. Deduplicate (MinHash LSH, threshold 0.85)
5. Tokenize with Nemotron tokenizer
6. Add language tags for multilingual training

### 5.2 Fine-tuning Data (500K+ pairs)
| Source | Size | Type |
|--------|------|------|
| **IndoMMLU** | 14K questions | Academic benchmark |
| **NusaX** | 12K sentences | Sentiment (12 languages) |
| **IndoNLI** | 100K pairs | Natural language inference |
| **Custom Instructions** | 400K+ pairs | Government/enterprise/health |

### 5.3 Local Language Coverage
| Language | Code | Speakers | Dataset |
|----------|------|----------|---------|
| Indonesian | id | 280M | Primary (all sources) |
| Javanese | jv | 98M | Wikipedia, NusaX |
| Sundanese | su | 42M | Wikipedia, NusaX |
| Balinese | ban | 4M | NusaX |
| Minangkabau | min | 6M | NusaX |
| Buginese | bug | 4M | Limited |
| Acehnese | ace | 3.5M | Limited |
| Banjarese | bjn | 3.5M | Limited |
| Ngaju Dayak | nij | 700K | Limited |
| Madurese | mad | 14M | Limited |

---

## 6. Benchmarking & Evaluation

### 6.1 Primary Benchmark: IndoMMLU
| Model | Accuracy | Notes |
|-------|----------|-------|
| Sahabat AI 8B | ~45% | Baseline |
| Sahabat AI 70B | ~52% | Current best Indonesian |
| SEA-LION v3-9B | ~55% | AI Singapore (SEA multilingual) |
| GPT-4 (zero-shot) | ~55% | General model |
| **Nemotron-Indonesia 30B** | **Target: 55%+** | **Our goal** |

### 6.2 Secondary Benchmarks
- **SEA-HELM / BHASA:** SEA-LION's benchmark — target competitive scores
- **NusaX Sentiment:** 12 local languages, target 80%+ accuracy
- **IndoNLI:** Natural language inference, target 75%+ accuracy
- **IndoSum:** Summarization, target ROUGE-L 35+
- **Custom Agentic Eval:** Tool use, multi-step reasoning (custom benchmark)

### 6.3 Evaluation Methodology
1. Run automated benchmarks (IndoMMLU, NusaX, IndoNLI, SEA-HELM)
2. Human evaluation on 100 diverse prompts
3. Compare against Sahabat AI 70B and SEA-LION v3 side-by-side
4. Safety evaluation (red-teaming for harmful outputs)
5. Agentic capability testing (function calling accuracy)

### 6.4 AI Singapore Collaboration Opportunity
SEA-LION's BHASA/SEA-HELM benchmark is the gold standard for SEA language evaluation. Nemotron-Indonesia should:
- Submit to SEA-LION Leaderboard for official ranking
- Propose joint research on agentic capabilities for SEA languages
- Share IndoMMLU training data (train split only) with AI Singapore community
- Explore NVIDIA + AI Singapore + Jatevo three-way partnership

---

## 7. Deployment Architecture

### 7.1 Inference Stack
```
Deployment Pipeline:
├── Model: Nemotron-Indonesia-30B-DPO (BF16)
├── Quantization: 4-bit (AWQ or GPTQ) for production
├── Serving: vLLM + Triton Inference Server
├── API: RESTful endpoint with OpenAI-compatible format
└── Monitoring: TensorRT-LLM performance metrics
```

### 7.2 Deployment Targets
| Environment | Spec | Purpose |
|-------------|------|---------|
| **Development** | 2× H200 | Testing and experimentation |
| **Staging** | 4× H200 | Pre-production validation |
| **Production** | 8× H200 | Full API serving |

### 7.3 API Specification
```
POST /v1/chat/completions
{
  "model": "nemotron-indonesia-30b",
  "messages": [
    {"role": "system", "content": "You are a helpful Indonesian AI assistant."},
    {"role": "user", "content": "Jelaskan konsep kecerdasan buatan dalam Bahasa Indonesia."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

---

## 8. Cost Analysis

### 8.1 Training Cost (Owned Hardware)
| Stage | Time | Cloud Equivalent | Actual Cost |
|-------|------|-----------------|-------------|
| Data preparation | 1 day | $500 | $10 (electricity) |
| Pre-training | 3 days | $5,000 | $30 (electricity) |
| SFT | 12 hours | $1,000 | $5 (electricity) |
| DPO | 6 hours | $500 | $3 (electricity) |
| **Total** | **~5 days** | **~$7,000** | **~$50** |

### 8.2 Inference Cost (Owned Hardware)
- **8× H200 idle:** ~$0 (sunk cost)
- **Electricity per day:** ~$10
- **Requests per day:** 1M+ (with vLLM optimization)
- **Cost per 1K requests:** ~$0.01

### 8.3 Cloud Comparison (if renting)
- **Lambda 8× H200:** ~$8/hour = ~$960/day
- **Full training:** ~$5,000
- **Monthly inference:** ~$6,000

---

## 9. Risk Assessment

### 9.1 Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| OOM during training | Medium | High | DeepSpeed ZeRO-3 + gradient checkpointing |
| Data quality issues | Medium | Medium | Rigorous cleaning + filtering pipeline |
| Underperforming benchmark | Low | High | Iterative evaluation + hyperparameter tuning |
| NVLink/InfiniBand failure | Low | High | Redundant network paths + monitoring |
| Checkpoint corruption | Low | High | Multiple save points + checksum validation |

### 9.2 Strategic Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Sahabat AI releases better model | Medium | Medium | Differentiation via agentic capabilities |
| Government policy changes | Low | Medium | Open source ensures community continuity |
| GPU hardware failure | Low | High | Spare GPU inventory + cloud fallback |
| Talent/competition | Medium | Medium | First-mover in agentic Indonesian LLM |

---

## 10. Timeline & Milestones

### 10.1 8-Week Roadmap
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| **1-2** | Data Curation | 20B token corpus ready |
| **3** | Infra Setup | Cluster configured, test run successful |
| **4** | Pre-training | Base model checkpoint (30B-Indo) |
| **5** | SFT | Instruction-tuned model |
| **6** | DPO | Aligned final model |
| **7** | Evaluation | Benchmark report vs Sahabat AI |
| **8** | Deployment | API endpoint + demo live |

### 10.2 Critical Path
1. **Data quality is #1 priority** — garbage data = garbage model
2. **Cluster stability** — 3-day training run must not crash
3. **Checkpoint frequency** — save every 500 steps minimum
4. **Validation during training** — run IndoMMLU every 1000 steps

---

## 11. Team & Resources

### 11.1 Required Team
| Role | Responsibility |
|------|---------------|
| **ML Engineer** | Training pipeline, DeepSpeed config, debugging |
| **Data Engineer** | Dataset curation, cleaning, quality validation |
| **DevOps Engineer** | Cluster setup, monitoring, deployment |
| **Research Scientist** | Evaluation design, benchmark analysis |
| **Product Manager** | Stakeholder communication, roadmap |

### 11.2 Current Resources
- **Hardware:** 8× H200 (owned by Jatevo)
- **Code:** Training pipeline ready at `~/projects/nemotron-indonesia/`
- **Data:** Pipeline to download and process all datasets
- **Cloud:** Optional fallback to Lambda/Vast.ai if needed

---

## 12. Deliverables

### 12.1 Immediate (This Week)
- [x] Training pipeline codebase
- [x] DeepSpeed ZeRO-3 configuration
- [x] Data preparation scripts
- [x] Architecture/PRD webpage
- [ ] Run test training (100 steps)
- [ ] Validate cluster connectivity

### 12.2 Short-term (Weeks 1-4)
- [ ] 20B token corpus prepared
- [ ] Pre-training completed
- [ ] First IndoMMLU evaluation
- [ ] Checkpoint validation

### 12.3 Medium-term (Weeks 5-8)
- [ ] SFT + DPO completed
- [ ] Full benchmark suite run
- [ ] Model quantized for inference
- [ ] API endpoint deployed

### 12.4 Long-term (Post-launch)
- [ ] Community feedback integration
- [ ] Iterative model improvements
- [ ] Enterprise partnerships
- [ ] Scale to 70B if needed

---

## 13. Appendices

### A. DeepSpeed ZeRO-3 Configuration
```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "nvme", "nvme_path": "/tmp/nvme_offload"},
    "offload_param": {"device": "nvme", "nvme_path": "/tmp/nvme_offload"},
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

### B. Training Command
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  train_nemotron_indonesia.py \
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

### C. References
- [Ilmu-Nemo-30B Announcement](https://theleaders-online.com/ytl-ai-labs-teams-up-with-nvidia-to-launch-ilmu%e2%80%91nemo%e2%80%9130b)
- [NVIDIA Nemotron Collection](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3)
- [Sahabat AI](https://sahabat-ai.com/)
- [Awesome Indonesian LLM Dataset](https://github.com/irfanfadhullah/awesome-indonesia-llm-dataset)

---

## 14. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Apr 23, 2026 | Jatevo AI | Initial PRD + architecture page |

**Next Review:** After pre-training completion (Week 4)

---

**Questions or feedback?** Contact: luca@jatevo.ai
