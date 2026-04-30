# Nemotron-Indonesia Omni 30B-A3B — Product Requirements Document

**Version:** 2.0
**Date:** April 29, 2026
**Status:** Repo pivoted before training; baseline evaluation next
**Owner:** Luca Cada Lora / Jatevo
**Architecture Page:** https://diagrams.jatevo.ai/nemotron-indonesia.html

---

## 1. Executive Summary

Nemotron-Indonesia is being updated from a text-only Indonesian LLM plan into **Nemotron-Indonesia Omni**: a sovereign multimodal agent model for Indonesian enterprises, built on NVIDIA's newly released **Nemotron 3 Nano Omni 30B-A3B Reasoning** model.

The timing is ideal because no training has started. We can pivot the base model before spending compute or creating checkpoints.

The new product thesis is stronger than the original: Indonesia does not only need another chat model. Enterprises need AI agents that can understand Bahasa Indonesia and local languages while also reading PDFs, scanned documents, tables, charts, screenshots, call audio, meeting recordings, and videos. Nemotron 3 Nano Omni provides that foundation.

### Base Model

| Purpose | Model |
|---|---|
| Flagship training/adaptation | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| Inference experiments | FP8 / NVFP4 Omni variants |
| Text benchmark fallback | Nemotron 3 Nano text 30B-A3B family if Omni underperforms on Indonesian text |

### Critical Caveat

NVIDIA's model card currently states **Language support: English only**. Nemotron-Indonesia Omni must therefore earn Indonesian capability through continued pretraining, SFT, multimodal SFT, and evaluation. We should not claim Indonesian-language production readiness until benchmarks validate it.

---

## 2. Product Goals

### Primary Goals

1. Build **Indonesia's first serious sovereign multimodal AI agent model**.
2. Adapt Nemotron 3 Nano Omni for Bahasa Indonesia and 10+ Indonesian local languages.
3. Enable enterprise workflows across text, document images, scanned PDFs, charts, tables, screenshots, audio, and video.
4. Preserve the original benchmark ambition: beat or match Sahabat AI 70B on IndoMMLU with a 30B-A3B active MoE model.
5. Deploy locally/sovereign using NVIDIA inference infrastructure: vLLM, TensorRT-LLM, and NIM-compatible services.

### Success Metrics

| Area | Metric | Target |
|---|---|---:|
| Indonesian reasoning | IndoMMLU accuracy | 55%+ |
| Local languages | NusaX / multilingual sentiment | 80%+ |
| NLI | IndoNLI | 75%+ |
| Summarization | IndoSum ROUGE-L | 35+ |
| Document intelligence | Indonesian scanned doc QA | 85%+ |
| Tables/charts | Field extraction accuracy | 90%+ |
| Audio | Bahasa audio QA semantic accuracy | 85%+ |
| Screenshots/UI | UI state reasoning accuracy | 80%+ |
| Agentic tools | Valid tool-call / JSON schema | 90%+ |
| Deployment | Local inference stack | vLLM / TensorRT-LLM |

---

## 3. Competitive Positioning

### 3.1 Sahabat AI

- **Builder:** Indosat Ooredoo Hutchison + GoTo
- **Base:** Llama/Gemma family
- **Focus:** General Indonesian chat
- **Strength:** Indonesian first-mover and ecosystem support
- **Gap:** Not positioned as a multimodal enterprise agent model

### 3.2 SEA-LION

- **Builder:** AI Singapore
- **Focus:** Southeast Asian multilingual text models and benchmarks
- **Strength:** Strong SEA language evaluation ecosystem
- **Gap:** Not Indonesia-specific and not primarily multimodal/agentic

### 3.3 Ilmu-Nemo

- **Builder:** YTL AI Labs × NVIDIA
- **Focus:** Malaysian agentic text model
- **Strength:** NVIDIA-aligned sovereign model story
- **Gap:** Malaysia-focused and text-first

### 3.4 Nemotron-Indonesia Omni

| Attribute | Sahabat AI | SEA-LION | Ilmu-Nemo | Nemotron-Indonesia Omni |
|---|---|---|---|---|
| Base | Llama / Gemma | MPT/Llama/Gemma | Nemotron text | **Nemotron 3 Nano Omni** |
| Mode | Text chat | SEA text | Agentic text | **Text + image + audio + video** |
| Context | Standard | Standard | Standard | **Up to 256k** |
| Indonesia-specific | Yes | Partial | No | **Yes** |
| Enterprise docs/OCR | Limited | Limited | Limited | **Core use case** |
| Tool use | Limited | Limited | Yes | **Core use case** |
| Sovereign deploy | Possible | Possible | NVIDIA stack | **NVIDIA stack** |

---

## 4. Base Model Technical Profile

### Nemotron 3 Nano Omni 30B-A3B Reasoning

| Attribute | Details |
|---|---|
| Architecture | Mamba2-Transformer Hybrid MoE |
| Parameters | ~31B total, 30B-A3B active MoE |
| Inputs | Text, image, video, audio |
| Output | Text |
| Context | Up to 256k tokens |
| Vision encoder | C-RADIOv4-H |
| Audio encoder | Parakeet |
| Video processing | 3D convolution + Efficient Video Sampling |
| Weights | BF16, FP8, NVFP4 |
| License | NVIDIA Open Model Agreement |
| Official runtime | vLLM, TensorRT-LLM, SGLang, NeMo, Megatron-Bridge |

### Official Sources

- HF BF16: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- HF FP8: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
- HF NVFP4: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
- NVIDIA cookbooks: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni
- Megatron-Bridge example: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/nemotron_3_omni
- DataDesigner long-doc recipes: https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes/vlm_long_doc

---

## 5. Technical Architecture

### 5.1 High-Level System

```text
Nemotron-Indonesia Omni
├── Base: NVIDIA Nemotron 3 Nano Omni 30B-A3B Reasoning
├── Indonesian text adaptation
│   ├── 20B curated Indonesian/local-language tokens
│   ├── benchmark decontamination
│   └── education-heavy curriculum
├── Multimodal adaptation
│   ├── Indonesian PDFs/scans/forms
│   ├── chart/table/document QA
│   ├── Bahasa audio + meeting QA
│   ├── Indonesian video/screen reasoning
│   └── GUI/browser/tool-use traces
├── Alignment
│   ├── DPO / GRPO
│   ├── grounded document answers
│   ├── safe enterprise refusals
│   └── structured tool calling
└── Deployment
    ├── vLLM OpenAI-compatible endpoint
    ├── TensorRT-LLM / NIM optimized endpoint
    └── enterprise agent demos
```

### 5.2 Hardware Infrastructure

```text
Target cluster:
├── 8× NVIDIA H200, 141GB VRAM each
├── Total VRAM: 1,128GB
├── NVLink / NVSwitch preferred
├── High-throughput NVMe for data and checkpoints
└── Optional cloud fallback: H100/H200/B200 instances
```

NVIDIA's Megatron-Bridge examples for Omni are verified on **8× H100 80GB nodes**. H200 should be sufficient, but we must validate memory on our exact training path.

### 5.3 Software Stack

| Component | Role |
|---|---|
| NeMo 26.04 container | Official Day-0 training/conversion environment |
| Megatron-Bridge `nemotron_3_omni` branch | Omni model conversion/training examples |
| Megatron-LM `nemotron_3_omni` branch | Model-parallel backend |
| vLLM 0.20.0 | OpenAI-compatible serving and smoke tests |
| TensorRT-LLM | Production inference optimization |
| NeMo DataDesigner | Synthetic long-document QA generation |
| NeMo Curator | Video/data processing |
| NeMo Evaluator | Benchmark/evaluation framework |

---

## 6. Training Strategy

### Phase 0 — Baseline Evaluation Before Training

Goal: measure upstream Omni before spending H200 time.

Tests:
- IndoMMLU 500–2,000 sample smoke test
- Indonesian PDF/table/chart QA
- scanned page OCR QA
- Indonesian audio QA/transcription sanity set
- UI/screenshot reasoning in Bahasa
- JSON/tool-call compliance

Output:
- `baseline_omni_report.md`
- Go/no-go for full adaptation
- Priority list of failure modes

### Phase 1 — Indonesian Text Continued Pretraining

Purpose: teach Bahasa Indonesia, local-language distribution, and Indonesian domain knowledge.

- **Input:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- **Data:** 20B curated Indonesian/local-language tokens
- **Domains:** education, government, news, law, finance, religion/culture, STEM, forums
- **Quality:** benchmark decontamination, deduplication, NER/entity density, spelling validation, KBBI checks
- **Output:** `nemotron-indonesia-omni-30b-pretrain`

### Phase 2 — Indonesian SFT

Purpose: make the model useful as an Indonesian assistant and reasoning model.

- IndoMMLU train split only
- NusaX / IndoNLI / IndoSum
- Government/enterprise FAQ
- Indonesian CoT/reasoning tasks
- Coding/tool-use tasks in Indonesian
- Output: `nemotron-indonesia-omni-30b-sft`

### Phase 3 — Multimodal SFT

Purpose: unlock the real differentiation.

Training data families:
- Indonesian PDFs rendered to page images
- scanned documents and OCR corrections
- tables, invoices, forms, contracts, procurement docs
- IDX/OJK/BI/company reports with chart/table QA
- call-center audio, meeting audio, and transcripts
- Indonesian screen/browser-agent tasks
- video QA from training/event/CCTV-style clips where legally allowed

Output:
- `nemotron-indonesia-omni-30b-mm-sft`

### Phase 4 — Preference / GRPO Alignment

Purpose: make the model reliable enough for enterprise agents.

Preference dimensions:
- grounded answers over documents
- correct refusal when source evidence is insufficient
- structured JSON/tool-call validity
- safe handling of private/sensitive documents
- concise Bahasa executive answers
- robust OCR/table correction

Output:
- `nemotron-indonesia-omni-30b-final`

### Phase 5 — Deployment

- BF16 for quality testing
- FP8/NVFP4 for inference cost and throughput
- vLLM endpoint first
- TensorRT-LLM/NIM endpoint for production
- enterprise demos: document agent, meeting agent, compliance agent, browser/screen agent

---

## 7. Data Strategy Summary

### 7.1 Text Corpus

| Source | Target Tokens | Purpose |
|---|---:|---|
| Indo4B HF mirror | 5.0B | core Indonesian formal + colloquial corpus |
| SEA-PILE Indonesian | 2.0B | curated SEA corpus |
| mC4 Indonesian filtered | 3.0B | web language breadth |
| CC100 Indonesian filtered | 2.5B | web diversity |
| Indonesian Wikipedia | 2.0B | structured knowledge |
| Indonesian academic/textbook corpus | 2.0B | IndoMMLU/STEM strength |
| News corpus | 2.0B | formal current language |
| Government/legal docs | 1.0B | civic/legal/domain knowledge |
| Local languages | 1.0B | Javanese/Sundanese/etc. |
| English STEM transfer | 0.5B | technical reasoning |
| **Total** | **20.0B** | |

### 7.2 Multimodal Corpus

| Data family | Initial target | Use case |
|---|---:|---|
| Rendered Indonesian PDFs | 500k–2M pages | document QA, OCR, tables |
| Scanned docs/forms | 100k–500k pages | noisy OCR robustness |
| Tables/charts/reports | 50k–200k examples | finance, BI, board docs |
| Audio + transcript + QA | 1k–10k hours | calls, meetings, voice notes |
| Video + transcript + QA | 500–2k hours | training/event/screen reasoning |
| Screenshot/UI traces | 10k–100k tasks | browser/GUI agents |
| Tool-call traces | 50k–200k tasks | enterprise agent reliability |

Full multimodal plan lives in `OMNI_DATA_STRATEGY.md`.

---

## 8. Evaluation Plan

### 8.1 Text Benchmarks

- IndoMMLU
- SEA-HELM / BHASA
- NusaX
- IndoNLI
- IndoSum
- Human eval: 100–300 Indonesian prompts across domains

### 8.2 Omni Benchmarks

Custom Indonesian benchmark pack:

1. **IndoDocQA** — PDFs, scans, reports, contracts, tables.
2. **IndoChartQA** — Indonesian financial/government charts.
3. **IndoAudioQA** — Bahasa calls/meetings with timestamps.
4. **IndoScreenQA** — screenshots and UI state reasoning.
5. **IndoAgentTools** — JSON/function calling and multi-step tool execution.

### 8.3 Acceptance Gates

| Gate | Requirement |
|---|---|
| Baseline | Upstream Omni must run locally/API without infrastructure blocker |
| Text | SFT model improves IndoMMLU vs upstream baseline |
| Multimodal | MM-SFT improves doc/audio/chart QA vs upstream baseline |
| Safety | No critical data leakage or unsafe document behavior in red-team eval |
| Deployment | vLLM/TensorRT endpoint sustains target latency under load |

---

## 9. Deployment Architecture

```text
Client / enterprise app
        ↓
OpenAI-compatible API gateway
        ↓
Nemotron-Indonesia Omni serving layer
├── vLLM BF16/FP8 endpoint for validation
├── TensorRT-LLM/NIM endpoint for production
├── retrieval/document parser services
├── audio/video preprocessing services
└── observability + eval logging
```

API model names:

```text
nemotron-indonesia-omni-30b-preview
nemotron-indonesia-omni-30b-final
nemotron-indonesia-omni-30b-nvfp4
```

Example request:

```json
{
  "model": "nemotron-indonesia-omni-30b-preview",
  "messages": [
    {"role": "system", "content": "Anda adalah asisten AI enterprise Indonesia yang teliti dan berbasis bukti."},
    {"role": "user", "content": "Ringkas dokumen ini dan ekstrak risiko utamanya."}
  ],
  "temperature": 0.2,
  "max_tokens": 2048
}
```

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Upstream model English-only | High | Phase 0 eval + Indonesian continued pretraining before claims |
| Multimodal training complexity | High | Start with text adaptation + doc QA SFT before full video/audio training |
| License ambiguity | Medium | Treat derivative release as subject to NVIDIA Open Model Agreement review |
| H200 memory/OOM | Medium | Begin with LoRA/adapters and smaller context; use Megatron-Bridge path |
| Data quality issues | High | KBBI/spell checks, benchmark decontamination, NER density, human spot checks |
| Benchmark contamination | High | 5/8/13-gram decontamination and train/test separation |
| Sensitive enterprise data | High | Use synthetic/permissioned data only; no private docs in public release |

---

## 11. 8-Week Roadmap

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Baseline Omni evaluation | baseline report + failure matrix |
| 1–2 | Data curation v2 | text + multimodal seed datasets |
| 2 | Infrastructure validation | vLLM + Megatron-Bridge smoke tests |
| 3–4 | Indonesian text adaptation | pretrain checkpoint + IndoMMLU delta |
| 5 | Indonesian SFT | SFT checkpoint + benchmark report |
| 6 | Multimodal doc/audio SFT | MM-SFT checkpoint + IndoDocQA report |
| 7 | DPO/GRPO alignment | final preview checkpoint |
| 8 | Deployment demo | API + document/audio/screen agent demos |

---

## 12. Immediate Deliverables

- [x] Review NVIDIA blog, HF model, and GitHub sources.
- [x] Update repo direction to Omni before training starts.
- [x] Add source review: `NEMOTRON_3_NANO_OMNI_GITHUB_REVIEW.md`.
- [x] Update PRD and README to v2 Omni direction.
- [ ] Run upstream Omni baseline tests.
- [ ] Validate vLLM 0.20.0 serving with BF16 or NVFP4.
- [ ] Build `IndoDocQA` seed eval set.
- [ ] Confirm NVIDIA Open Model Agreement implications for public derivative release.

---

## 13. References

- NVIDIA launch blog: https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/
- NVIDIA technical blog: https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model
- HF BF16: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- NVIDIA-NeMo Nemotron cookbooks: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni
- Megatron-Bridge Omni example: https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/nemotron_3_omni
- DataDesigner recipes: https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes/vlm_long_doc
- SEA-LION: https://huggingface.co/collections/aisingapore/sea-lionv3-672589a39cdadd6a5b199581
- Sahabat AI: https://sahabat-ai.com/

---

## 14. Document Control

| Version | Date | Changes |
|---|---|---|
| 1.0 | Apr 23, 2026 | Initial text-only Nemotron-Indonesia PRD |
| 2.0 | Apr 29, 2026 | Pivoted to Nemotron 3 Nano Omni flagship base before training |

**Next Review:** after upstream Omni baseline evaluation.
