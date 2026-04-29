# Nemotron 3 Nano Omni — GitHub / Base Model Review

**Date:** 2026-04-29  
**Question:** Should Nemotron-Indonesia use NVIDIA Nemotron 3 Nano Omni as the base model?

## Short Answer

Yes — but with a specific positioning: use **Nemotron 3 Nano Omni** as the base for **Nemotron-Indonesia Omni**, the multimodal/agentic version of the project. Do not blindly replace the text-only Indonesian LLM plan until we run Indonesian benchmark smoke tests, because NVIDIA's model card currently states **Language support: English only**.

Best path:

1. Keep the original text model path for pure Bahasa / local-language benchmark work.
2. Add a new multimodal track: **Nemotron-Indonesia Omni 30B-A3B**.
3. Fine-tune/adapt Omni on Indonesian document, audio, video, OCR, screen, and enterprise-agent data.
4. Evaluate before committing full training spend.

## Official Sources Checked

### Model / Weights

- Hugging Face BF16: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- Hugging Face FP8: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
- Hugging Face NVFP4: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
- NVIDIA build endpoint: https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
- NGC/NIM container referenced by model card.

### Official GitHub Repositories

- Main Nemotron cookbooks:  
  https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Nano-Omni

  Contains:
  - `vllm_cookbook.ipynb`
  - `trtllm_cookbook.ipynb`
  - `sglang_cookbook.ipynb`
  - `automodel/`
  - `doc-intelligence-with-parse/`
  - `grpo/`
  - `grpo_nemo_gym/`
  - `Megatron-bridge/`

- Megatron-Bridge model example:  
  https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/nemotron_3_omni

  Notes from README:
  - Nemotron-3 Nano Omni is a 30B-A3B MoE multimodal model.
  - Jointly processes image, video, audio, and text.
  - Uses MoE Mamba/attention hybrid language backbone, RADIO vision tower, and Parakeet sound encoder.
  - Verified flows on NVIDIA H100 80GB nodes with 8 GPUs per node.
  - Day-0 branch: `nemotron_3_omni`.
  - Base container: `nvcr.io/nvidia/nemo:26.04`.

- Megatron-LM branch referenced by Day-0 code:  
  https://github.com/NVIDIA/Megatron-LM/tree/nemotron_3_omni

- DataDesigner long-document synthetic data recipes:  
  https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes/vlm_long_doc

  Contains recipes for:
  - seed dataset preparation
  - OCR SDG with Nemotron Parse
  - text QA SDG
  - page classification SDG
  - visual QA SDG
  - single-page QA SDG
  - multi-page windowed QA SDG
  - whole-document QA SDG
  - frontier judge SDG

- Curator video tooling:  
  https://github.com/NVIDIA-NeMo/Curator/tree/main

- Evaluator:  
  https://github.com/NVIDIA-NeMo/Evaluator

## Model Facts

- Architecture: **Mamba2-Transformer Hybrid Mixture of Experts**
- Size: **~31B total / 30B-A3B active MoE**
- Modalities in: **video, audio, image, text**
- Output: **text**
- Context: up to **256k tokens**
- Vision encoder: **C-RADIOv4-H**
- Audio encoder: **Parakeet**
- Video: 3D convolution + Efficient Video Sampling
- Use cases: document intelligence, OCR, GUI/screen agents, meeting/video/audio understanding, enterprise agents
- Model card says: **Language support: English only**
- License: **NVIDIA Open Model Agreement**; HF reports `license: other`, not a standard OSI license, but model card says commercial use is available.

## Deployment Notes

- BF16 weights are about **62GB**.
- vLLM required version in model card: **0.20.0**.
- Supported runtimes: vLLM, TensorRT-LLM, TensorRT Edge-LLM, SGLang, llama.cpp, Ollama, NeMo, Megatron, NeMo-RL.
- Supported hardware listed: Ampere A100 80GB, Hopper H100/H200, Blackwell B200 / RTX Pro 6000 / DGX Spark / Jetson Thor / RTX 5090, Lovelace L40S.
- Megatron-Bridge examples say conversion/inference/training flows were verified on **8× H100 80GB nodes**.
- vLLM serving example requires `--trust-remote-code` and explicit video sampling settings.

## Suitability for Nemotron-Indonesia

### Strong Reasons to Use It

- It gives Nemotron-Indonesia an immediate multimodal angle: Indonesian documents, PDFs, slides, screenshots, CCTV/video, call/audio, meetings, and enterprise workflows.
- Agentic positioning becomes much stronger than Sahabat AI: not just chat, but perception + tools + document/video/audio understanding.
- 256k context is a major upgrade versus the original PRD's 4k context assumption.
- Official NVIDIA recipes exist for inference, conversion, GRPO, document intelligence, and synthetic document QA.
- Commercial-use path exists under NVIDIA Open Model Agreement.

### Major Caveats

- **English-only language support** means it is not plug-and-play for Bahasa Indonesia or local languages.
- Fine-tuning multimodal models is materially harder than text-only SFT.
- Requires careful license review before calling the final derivative model fully “open source.”
- BF16/FP8/NVFP4 deployment requires modern NVIDIA hardware and updated inference stack.
- PDF ingestion is image-page based, not raw PDF-native.

## Recommendation

Proceed with a two-track architecture:

### Track A — Nemotron-Indonesia Text

Purpose: win IndoMMLU / NusaX / Indonesian local-language benchmarks.

Base:
- Continue using the text Nemotron 3 Nano 30B-A3B base/instruct family unless Omni proves equal or better on Indonesian text.

Training:
- Indonesian continued pretraining
- Indonesian + local-language SFT
- DPO/GRPO for agentic tool use

### Track B — Nemotron-Indonesia Omni

Purpose: become Indonesia's first serious multimodal sovereign agent model.

Base:
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` for training/adaptation.
- `FP8` or `NVFP4` for inference/deployment experiments.

Training/adaptation data:
- Indonesian government PDFs and scanned documents
- IDX/OJK/BI/company reports, pages rendered to image
- Indonesian call-center audio and meeting transcripts
- Bahasa + local-language speech/audio pairs
- Indonesian UI/screenshot/browser-agent tasks
- enterprise workflow traces with function calls
- local charts, tables, invoices, forms, contracts, procurement docs

First proof point:
- Build a **Nemotron-Indonesia Omni Document Agent** that reads Indonesian PDFs, scans, tables, charts, and meeting/audio/video snippets.

## Immediate Next Steps

1. Run baseline eval of upstream Omni on:
   - IndoMMLU sample
   - Bahasa document OCR QA
   - Indonesian PDF/table extraction
   - Indonesian audio transcription
   - UI/screenshot reasoning

2. Create a small adaptation set:
   - 5k Indonesian doc QA
   - 2k scanned/OCR pages
   - 1k audio/transcription QA
   - 1k chart/table questions
   - 1k agent/tool-call tasks

3. Try LoRA/QLoRA or adapter-style SFT first before full continued pretraining.

4. If scores move, update the PRD to position the project as:

   **Nemotron-Indonesia Omni: sovereign multimodal AI agents for Indonesian enterprises.**
