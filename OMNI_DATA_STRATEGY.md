# Nemotron-Indonesia Omni — Multimodal Data Strategy

**Version:** 1.0  
**Date:** 2026-04-29  
**Purpose:** Adapt NVIDIA Nemotron 3 Nano Omni for Indonesian enterprise multimodal agents.

---

## 1. Strategy

The original Nemotron-Indonesia data plan focused on text benchmarks. That remains necessary, but Omni needs a second layer: Indonesian documents, images, audio, video, screenshots, and tool-use traces.

The goal is not just “Bahasa chat.” The goal is:

> Indonesian agents that can read, hear, see, reason, and act inside enterprise workflows.

---

## 2. Data Families

| Family | Initial Target | Priority | Use Case |
|---|---:|---|---|
| Indonesian text corpus | 20B tokens | P0 | language + knowledge adaptation |
| Rendered PDFs | 500k–2M pages | P0 | document QA, scanned reports, tables |
| OCR correction pairs | 100k–500k pages | P0 | robust Indonesian OCR reasoning |
| Tables/charts | 50k–200k examples | P0 | finance, BI, board docs, government stats |
| Audio + transcript | 1k–10k hours | P1 | calls, meetings, voice notes |
| Video + transcript | 500–2k hours | P1 | training/event/screen understanding |
| Screenshot/UI tasks | 10k–100k tasks | P1 | browser/GUI agents |
| Tool-call traces | 50k–200k tasks | P0 | agent reliability and structured output |
| Safety/red-team docs | 5k–20k examples | P0 | privacy, refusal, sensitive document handling |

---

## 3. Phase 0 Baseline Eval Set

Before training, build a small but representative eval set to measure upstream Omni.

### 3.1 IndoText Smoke

- 500–2,000 IndoMMLU examples.
- 100 local-language prompts.
- 100 Indonesian enterprise Q&A prompts.

### 3.2 IndoDocQA Seed

- 100 PDF pages from public Indonesian documents.
- 50 scanned/noisy pages.
- 50 tables/charts.
- Questions should require evidence from the page, not memorization.

Example item:

```json
{
  "id": "indodocqa_0001",
  "modality": "image+text",
  "image": "pages/ojk_report_001.png",
  "question": "Berapa total aset yang dilaporkan pada tabel utama?",
  "answer": "Rp ...",
  "evidence_bbox": [120, 340, 880, 420],
  "source": "public"
}
```

### 3.3 IndoAudioQA Seed

- 10–20 hours of public/permissioned Indonesian speech.
- Include accents, formal/informal speech, noisy audio.
- Pair with transcript and Q&A.

### 3.4 IndoScreenQA Seed

- Screenshots of Indonesian web apps/forms/dashboards.
- Browser-agent tasks in Bahasa.
- Must include expected UI state/action.

---

## 4. Training Data Formats

### 4.1 Text-only Continued Pretraining

```text
<|id|>
{clean Indonesian text}
```

Use existing `DATA_STRATEGY.md` for quality filtering, deduplication, decontamination, and curriculum sorting.

### 4.2 Document QA

Use rendered page images plus structured prompt/answer/evidence.

```json
{
  "messages": [
    {"role": "system", "content": "Jawab berdasarkan dokumen. Jika tidak ada bukti, katakan tidak ditemukan."},
    {"role": "user", "content": [
      {"type": "image", "image": "page_001.png"},
      {"type": "text", "text": "Apa nilai EBITDA pada tabel ini?"}
    ]},
    {"role": "assistant", "content": "Nilai EBITDA adalah ... berdasarkan tabel pada halaman 1."}
  ],
  "metadata": {
    "language": "id",
    "source": "public",
    "task": "table_qa"
  }
}
```

### 4.3 Audio QA

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "audio", "audio": "meeting_001.wav"},
      {"type": "text", "text": "Apa keputusan utama dalam rapat ini?"}
    ]},
    {"role": "assistant", "content": "Keputusan utama adalah ..."}
  ],
  "metadata": {
    "language": "id",
    "task": "meeting_summary",
    "has_transcript": true
  }
}
```

### 4.4 Tool-Call / Agent Traces

```json
{
  "messages": [
    {"role": "system", "content": "Gunakan tool jika perlu. Output harus valid JSON."},
    {"role": "user", "content": "Cari invoice yang jatuh tempo minggu ini dan buat ringkasan."},
    {"role": "assistant", "tool_calls": [
      {"name": "search_documents", "arguments": {"query": "invoice jatuh tempo minggu ini"}}
    ]}
  ],
  "metadata": {
    "language": "id",
    "task": "tool_call"
  }
}
```

---

## 5. Quality Gates

| Gate | Rule |
|---|---|
| Source permission | Only public, licensed, synthetic, or explicitly permissioned data |
| PII | Remove or mask personal identifiers unless data is synthetic/test-only |
| OCR quality | Track raw OCR, corrected OCR, and source image separately |
| Evidence | Document QA should include page/evidence pointer where possible |
| Decontamination | Remove eval/test overlap before training |
| Language quality | KBBI/spell checks for Indonesian outputs; local-language panel for Javanese/Sundanese/etc. |
| Hallucination | Include “not found in document” examples |
| Tool validity | Validate JSON schema automatically |

---

## 6. Synthetic Data Generation

Use NVIDIA DataDesigner recipes as starting point:

- OCR SDG
- text QA SDG
- page classification SDG
- visual QA SDG
- single-page QA
- multi-page windowed QA
- whole-document QA
- frontier judge SDG

Official source:
https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes/vlm_long_doc

Indonesian adaptation:
- generate questions in Bahasa Indonesia,
- require answer evidence from source page,
- include formal and informal question styles,
- include “jawaban tidak tersedia di dokumen” negatives,
- judge answer faithfulness against source text/image.

---

## 7. First Practical Dataset Build

### Week 1 Target

Build a small seed pack:

| Dataset | Size |
|---|---:|
| IndoMMLU smoke | 1,000 questions |
| Indonesian PDF pages | 1,000 pages |
| OCR/doc QA | 5,000 Q&A pairs |
| Chart/table QA | 1,000 Q&A pairs |
| Audio QA | 50 hours or equivalent public/permissioned audio |
| Screenshot/UI QA | 1,000 tasks |
| Tool-call tasks | 2,000 traces |

This is enough to answer the most important question: does Omni move after small Indonesian adaptation?

---

## 8. Recommended Training Order

1. Baseline eval upstream Omni.
2. Indonesian text continued pretraining or LoRA/adapters.
3. SFT on Indonesian instructions and benchmark train splits.
4. Document/OCR/chart multimodal SFT.
5. Audio/screenshot/tool-call SFT.
6. Preference/GRPO alignment.
7. Quantized inference validation.

---

## 9. Non-Negotiables

- Do not train on private enterprise documents without explicit permission.
- Do not mix benchmark test sets into training.
- Do not claim open-source final weights until NVIDIA Open Model Agreement review is complete.
- Do not claim Indonesian production quality until baseline and post-training evals prove it.
