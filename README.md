# Nemotron-Indonesia Omni

**Indonesian multimodal/agentic AI model adaptation built on NVIDIA Nemotron 3 Nano Omni.**

This repository contains the public training architecture, data-source manifest, and scaffolding for adapting `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` for Bahasa Indonesia and Indonesian enterprise-style tasks.

No client-specific project references, private project-management integrations, or production credentials belong in this repository.

---

## Start Here for the Training Operator

If you are setting up the training machine, start with the operator runbook:

- [`OPERATOR_RUNBOOK.md`](OPERATOR_RUNBOOK.md)

Minimal flow:

```bash
git clone https://github.com/lucacadalora/nemotron-indonesia.git
cd nemotron-indonesia
bash START_HERE.sh
python download_sources.py --sources first_milestone
```

`first_milestone` pulls only the initial sources needed to validate the pipeline: Indo4B HF mirror, Indonesian Wikipedia, IndoNLU, and IndoBERT.

---

## Base Model

| Track | Role | Base |
|---|---|---|
| Flagship | Indonesian multimodal / agentic model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| Inference variants | Lower-cost serving tests | FP8 / NVFP4 Omni variants |
| Text fallback | Pure language benchmark experiments | Nemotron 3 Nano text 30B-A3B family |

Official NVIDIA sources:

- BF16 model: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- FP8 model: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
- NVFP4 model: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
- NVIDIA blog: https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model
- Nemotron cookbooks: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni
- Megatron-Bridge Omni example: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/nemotron_3_omni

Important caveat: NVIDIA's upstream model card states English-only language support. This project exists to adapt and evaluate Indonesian capability.

---

## Concrete Architecture

Read the full architecture here:

- [`ARCHITECTURE.md`](ARCHITECTURE.md)

Short version:

```text
Indo4B / SEA-PILE ID / mC4 ID / CC100 ID / Wikipedia ID
        ↓
cleaning + language filtering + dedupe + PII removal + benchmark decontamination
        ↓
continued pretraining / domain adaptation
        ↓
Nemotron-Indonesia base
        ↓
Bahasa + enterprise SFT + multimodal SFT
        ↓
alignment / preference tuning
        ↓
IndoNLU + NusaX + IndoBERT baseline + custom enterprise benchmark
```

---

## Data Sources and Download Links

Read the source manifest here:

- [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md)

The repo does **not** depend on manually gated OSCAR. Core pull list:

| Key | Source | Link |
|---|---|---|
| `indo4b_hf` | Indo4B HF parquet mirror | https://huggingface.co/datasets/taufiqdp/Indo4B-hf |
| `sea_pile_id` | SEA-PILE Indonesian subset | https://huggingface.co/datasets/aisingapore/SEA-PILE-v1 |
| `mc4_id` | mC4 Indonesian | https://huggingface.co/datasets/allenai/c4 |
| `cc100_id` | CC100 Indonesian | https://data.statmt.org/cc-100/id.txt.xz |
| `wikipedia_id` | Indonesian Wikipedia | https://huggingface.co/datasets/wikimedia/wikipedia |
| `indonlu` | IndoNLU benchmark | https://huggingface.co/datasets/indonlp/indonlu |
| `nusax_senti` | NusaX sentiment benchmark | https://huggingface.co/datasets/indonlp/NusaX-senti |
| `indobert` | IndoBERT baseline model | https://huggingface.co/indobenchmark/indobert-base-p1 |

Download helper:

```bash
python download_sources.py --list
python download_sources.py --sources core --dry-run
python download_sources.py --sources core
```

Small first pull for pipeline validation:

```bash
python download_sources.py --sources first_milestone --dry-run
python download_sources.py --sources first_milestone
```

Optional CulturaX Indonesian pull after accepting Hugging Face terms:

```bash
HF_TOKEN=hf_xxx python download_sources.py --sources culturax_id
```

---

## IndoBERT and IndoNLU Usage

IndoBERT and IndoNLU are important, but not as a base model replacement.

| Asset | Role in this project |
|---|---|
| IndoBERT | baseline/reference model, quality-filter helper, comparison point |
| IndoNLU | benchmark/evaluation suite and small supervised seed from non-test splits |
| Indo4B | larger Indonesian text corpus suitable for continued pretraining |

Rule: use IndoNLU for evaluation and decontamination. Do not leak benchmark test examples into training.

---

## Repository Structure

```text
nemotron-indonesia/
|-- README.md
|-- ARCHITECTURE.md                             # Concrete public architecture
|-- DATASET_MANIFEST.md                         # Working source links + download commands
|-- PRD.md                                      # Product/technical PRD
|-- DATA_STRATEGY.md                            # Text data strategy
|-- OMNI_DATA_STRATEGY.md                       # Multimodal data strategy
|-- NEMOTRON_3_NANO_OMNI_GITHUB_REVIEW.md       # Source/model/GitHub review
|-- download_sources.py                         # One-command data/model pull helper
|-- prepare_data.py                             # Indonesian text curation pipeline scaffold
|-- train_nemotron_indonesia.py                 # Text-adaptation training scaffold
|-- run_training.sh                             # Launcher, defaults to Omni BF16
|-- evaluate.py                                 # Benchmark scaffold
|-- START_HERE.sh                               # Setup guide for H200 machine
|-- configs/
|   |-- pretrain.yaml
|   |-- sft.yaml
|   |-- dpo.yaml
|   |-- omni_multimodal_sft.yaml
|   |-- deepspeed_zero3.json
|   |-- deepspeed_zero3_gpuonly.json
|-- data/                                       # Generated / mounted training data
|-- models/                                     # Generated checkpoints
```

---

## Hardware / Runtime

Target training/inference hardware:

- 8× NVIDIA H200 preferred for full adaptation.
- H100 80GB is also viable for smaller experiments.
- BF16 weights are roughly 62GB.
- Upstream model card requirement for vLLM: `vLLM 0.20.0`.
- NeMo/Megatron-Bridge Day-0 base container: `nvcr.io/nvidia/nemo:26.04`.

Serving stack:

- vLLM for OpenAI-compatible testing.
- TensorRT-LLM / NIM-compatible serving for optimized deployment.
- FP8 / NVFP4 variants for inference experiments.

---

## Quick Start

### 1. Environment

```bash
conda create -n nemotron-indonesia python=3.10 -y
conda activate nemotron-indonesia
pip install -r requirements.txt
```

### 2. Validate base tokenizer

```bash
python - <<'PY'
from transformers import AutoTokenizer
model = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("tokenizer ok", len(tok))
PY
```

### 3. Pull first data sources

```bash
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert --dry-run
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert
```

### 4. Prepare Indonesian text data

```bash
python prepare_data.py \
  --output_dir ./data/processed \
  --datasets indo4b_hf wikipedia cc100 seapile \
  --tokenizer_name nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --use_ner_filter \
  --quality_threshold 0.1
```

### 5. Training scaffold

```bash
./run_training.sh pretrain
./run_training.sh sft
./run_training.sh dpo
```

The current Python trainer is a text-adaptation scaffold. Full multimodal training should follow NVIDIA's NeMo/Megatron-Bridge Omni examples.

### 6. Evaluation

```bash
python evaluate.py \
  --model_path ./models/nemotron-indonesia-omni-30b-dpo \
  --benchmark indonlu \
  --output results_indonlu.json
```

---

## Training Roadmap

| Phase | Purpose | Output |
|---|---|---|
| 0. Baseline smoke tests | Check upstream Omni on Indonesian text, docs, OCR, audio, screenshots | go/no-go report |
| 1. Indonesian text adaptation | Improve Bahasa + local-language text reasoning | CPT checkpoint |
| 2. Indonesian SFT | Assistant behavior, reasoning, enterprise workflows | SFT checkpoint |
| 3. Multimodal SFT | Indonesian docs, scans, charts, tables, audio/video QA | MM-SFT checkpoint |
| 4. Preference / GRPO | Tool use, reliability, doc grounding, refusal behavior | final aligned checkpoint |
| 5. Deployment | vLLM / TensorRT-LLM / NIM endpoint | local API + demos |

---

## License Note

The training code and recipes in this repo can be released openly by Jatevo, but the base model is governed by the NVIDIA Open Model Agreement. Final model release terms must be reviewed against NVIDIA's agreement before public distribution.

---

Built for Indonesia's AI sovereignty — now multimodal.
