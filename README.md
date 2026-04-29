# Nemotron-Indonesia Omni

**Sovereign multimodal AI agents for Indonesia, built on NVIDIA Nemotron 3 Nano Omni.**

Nemotron-Indonesia is now oriented around NVIDIA's new **Nemotron 3 Nano Omni 30B-A3B Reasoning** model as the flagship base. The project still keeps a text-benchmark track for IndoMMLU and local-language performance, but the primary product thesis is now bigger: Indonesian enterprise agents that can reason over **text, PDFs, scanned documents, charts, screenshots, audio, video, and tools**.

> Pivot note — 2026-04-29: training has not started, so the repo has been updated before any checkpoint is created.

---

## Base Model Decision

| Track | Role | Base |
|---|---|---|
| **Flagship: Omni** | Indonesian multimodal / agentic model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| **Inference variants** | Lower-cost serving tests | FP8 / NVFP4 Omni variants |
| **Text benchmark fallback** | Pure IndoMMLU / local-language work if Omni underperforms | Nemotron 3 Nano text 30B-A3B family |

Why Omni:
- 30B-A3B MoE, Mamba/Transformer hybrid.
- Inputs: text, image, audio, video.
- Output: text.
- Up to 256k context.
- Built for document intelligence, OCR, GUI agents, video/audio reasoning, and tool use.
- Official NVIDIA cookbooks exist for vLLM, TensorRT-LLM, SGLang, NeMo/Megatron-Bridge, GRPO, and document intelligence.

Important caveat:
- NVIDIA's model card currently states **English-only language support**. Nemotron-Indonesia must adapt it for Bahasa Indonesia and local languages before claiming Indonesian capability.

---

## Goals

1. **Build Nemotron-Indonesia Omni 30B-A3B** as Indonesia's first serious sovereign multimodal agent model.
2. **Beat or match Indonesian text benchmarks**: IndoMMLU target 55%+, NusaX 80%+.
3. **Support Indonesian enterprise modalities**: PDFs, scanned pages, tables, charts, forms, screenshots, meetings, audio calls, and videos.
4. **Support 10+ Indonesian languages**: Indonesian, Javanese, Sundanese, Balinese, Minangkabau, Buginese, Acehnese, Banjarese, Ngaju Dayak, Madurese.
5. **Deploy locally** using NVIDIA inference stack: vLLM / TensorRT-LLM / NIM-compatible serving.

---

## Repository Structure

```text
nemotron-indonesia/
|-- README.md
|-- PRD.md                                      # Updated v2 product/technical PRD
|-- DATA_STRATEGY.md                            # Text benchmark data strategy
|-- OMNI_DATA_STRATEGY.md                       # Multimodal Indonesian data strategy
|-- NEMOTRON_3_NANO_OMNI_GITHUB_REVIEW.md       # Source/model/GitHub review
|-- train_nemotron_indonesia.py                 # Current text-adaptation training scaffold
|-- run_training.sh                             # Launcher, now defaults to Omni BF16
|-- prepare_data.py                             # Indonesian text curation pipeline
|-- evaluate.py                                 # IndoMMLU / SEA-HELM benchmark scaffold
|-- multica_sync.py                             # Project tracking sync
|-- START_HERE.sh                               # Setup guide for H200 machine
|-- configs/
|   |-- pretrain.yaml                           # Omni-oriented text continued pretrain config
|   |-- sft.yaml                                # Indonesian SFT config
|   |-- dpo.yaml                                # Preference alignment config
|   |-- omni_multimodal_sft.yaml                # New multimodal SFT config scaffold
|   |-- deepspeed_zero3.json
|   |-- deepspeed_zero3_gpuonly.json
|-- data/                                       # Generated / mounted training data
|-- models/                                     # Generated checkpoints
```

---

## Official NVIDIA Sources

- HF BF16: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- HF FP8: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
- HF NVFP4: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
- NVIDIA technical blog: https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model
- Main cookbooks: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni
- Megatron-Bridge example: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/nemotron_3_omni
- DataDesigner long-doc recipes: https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes/vlm_long_doc

---

## Hardware / Runtime

Target training/inference hardware:
- **8× NVIDIA H200** preferred for full adaptation.
- H100 80GB also supported by NVIDIA examples.
- BF16 weights are ~62GB.
- vLLM model card requirement: **vLLM 0.20.0**.
- NeMo/Megatron-Bridge Day-0 base container: `nvcr.io/nvidia/nemo:26.04`.

Serving stack:
- vLLM for OpenAI-compatible testing.
- TensorRT-LLM / NIM for production optimization.
- FP8 / NVFP4 for inference experiments.

---

## Quick Start

### 1. Environment

```bash
conda create -n nemotron-indonesia python=3.10 -y
conda activate nemotron-indonesia
pip install -r requirements.txt
```

For official Omni Day-0 training/conversion flows, use NVIDIA's NeMo 26.04 container and Megatron-Bridge `nemotron_3_omni` branch as documented in the GitHub review.

### 2. Download / Validate Base Model

```bash
python - <<'PY'
from transformers import AutoTokenizer
model = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("tokenizer ok", len(tok))
PY
```

### 3. Prepare Indonesian Text Data

```bash
python prepare_data.py \
  --output_dir ./data/processed \
  --datasets oscar cc100 wikipedia sealion \
  --tokenizer_name nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --use_ner_filter \
  --quality_threshold 0.1
```

### 4. Text Adaptation Scaffold

```bash
./run_training.sh pretrain
./run_training.sh sft
./run_training.sh dpo
```

Current launcher defaults to the Omni BF16 model ID. The current Python trainer is still a text-adaptation scaffold; full multimodal training should follow NVIDIA's Megatron-Bridge/NeMo Omni examples.

### 5. Evaluation

```bash
python evaluate.py \
  --model_path ./models/nemotron-indonesia-omni-30b-dpo \
  --benchmark indommlu \
  --output results_indommlu.json
```

---

## Training Roadmap

| Phase | Purpose | Output |
|---|---|---|
| 0. Baseline smoke tests | Check upstream Omni on Indonesian text, docs, OCR, audio, screenshots | Go/no-go report |
| 1. Indonesian text adaptation | Improve Bahasa + local-language text reasoning | `nemotron-indonesia-omni-30b-pretrain` |
| 2. Indonesian SFT | IndoMMLU, NusaX, government/enterprise tasks | `nemotron-indonesia-omni-30b-sft` |
| 3. Multimodal SFT | Indonesian docs, scans, charts, tables, audio/video QA | `nemotron-indonesia-omni-30b-mm-sft` |
| 4. Preference / GRPO | Tool use, reliability, doc grounding, refusal behavior | `nemotron-indonesia-omni-30b-final` |
| 5. Deployment | vLLM / TensorRT-LLM / NIM endpoint | Enterprise API + demos |

---

## Benchmarks

### Text / Language

| Benchmark | Target |
|---|---:|
| IndoMMLU | 55%+ |
| NusaX sentiment / local languages | 80%+ |
| IndoNLI | 75%+ |
| IndoSum | ROUGE-L 35+ |

### Omni / Enterprise

| Benchmark | Target |
|---|---:|
| Indonesian scanned document QA | 85%+ answer accuracy |
| Indonesian table/chart extraction | 90%+ field accuracy |
| Bahasa audio transcription QA | 85%+ semantic accuracy |
| Screenshot / UI reasoning | 80%+ task-state accuracy |
| Tool-call correctness | 90%+ valid JSON/tool schema |

---

## Competitive Positioning

| Attribute | Sahabat AI | SEA-LION | Ilmu-Nemo | Nemotron-Indonesia Omni |
|---|---|---|---|---|
| Base | Llama / Gemma | MPT/Llama/Gemma | Nemotron text | **Nemotron 3 Nano Omni** |
| Primary mode | Chat | SEA multilingual | Agentic text | **Multimodal agents** |
| Modalities | Text | Text | Text | **Text + image + audio + video** |
| Indonesia-specific | Yes | Partial | No | **Yes** |
| Local languages | Limited | SEA-wide | Malay + EN | **10+ target** |
| Enterprise docs/OCR | No | Limited | Limited | **Core use case** |
| Sovereign deploy | Possible | Possible | NVIDIA stack | **NVIDIA local stack** |

---

## License Note

The training code and recipes in this repo can be released openly by Jatevo, but the base model is governed by the **NVIDIA Open Model Agreement**, not Apache 2.0. Final model release terms must be reviewed against NVIDIA's agreement before public distribution.

---

Built for Indonesia's AI sovereignty — now multimodal.
