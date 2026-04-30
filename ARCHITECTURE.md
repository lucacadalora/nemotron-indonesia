# Nemotron-Indonesia — Concrete Training Architecture

This repo adapts `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` into an Indonesian-capable multimodal/agentic model.

The architecture is intentionally public-safe: it contains no client-specific references, no private project-management integration, and no production credentials.

---

## 1. System Goal

Build a local Indonesian AI model that can:

1. Understand and generate high-quality Bahasa Indonesia.
2. Improve performance on Indonesian NLP/NLU benchmarks.
3. Handle enterprise-style Indonesian documents, tables, forms, screenshots, audio, and workflows.
4. Run on NVIDIA infrastructure with local deployment options.

---

## 2. Three-Layer Training Plan

```text
Layer 1 — Corpus / CPT
Indo4B + SEA-PILE ID + mC4 ID + CC100 ID + Wikipedia ID + optional CulturaX ID
        ↓
cleaning + language filtering + dedupe + PII removal + benchmark decontamination
        ↓
continued pretraining / domain-adaptive pretraining
        ↓
Nemotron-Indonesia base

Layer 2 — Instruction / SFT
Indonesian instructions + localized reasoning + enterprise workflow tasks
        ↓
SFT / multimodal SFT
        ↓
Nemotron-Indonesia instruct

Layer 3 — Evaluation / Proof
IndoNLU + NusaX + IndoBERT baselines + custom Indonesian enterprise benchmark
        ↓
scorecard against upstream Nemotron and Indonesian baselines
        ↓
release candidate
```

---

## 3. Layer 1 — Corpus Layer

Purpose: make the base model stronger in Indonesian before instruction tuning.

Primary corpus sources:

| Priority | Source | Role | Access |
|---:|---|---|---|
| 1 | Indo4B HF mirror | Indonesian formal + colloquial corpus, close to original IndoBERT training corpus | public |
| 2 | SEA-PILE Indonesian | curated Southeast Asian web corpus, Indonesian subset | public |
| 3 | mC4 Indonesian | large Common Crawl-derived Indonesian text | public |
| 4 | CC100 Indonesian | CC-Net/XLM-R Indonesian corpus | public |
| 5 | Indonesian Wikipedia | structured factual/encyclopedic knowledge | public |
| 6 | CulturaX Indonesian | large deduplicated Indonesian web corpus | optional; HF auto-gated |

Processing steps:

1. Normalize Unicode and whitespace.
2. Strip HTML, boilerplate, repeated navigation text, malformed JSON fragments.
3. Language-filter for Indonesian plus selected local languages.
4. Remove emails, phone numbers, personal identifiers where possible.
5. Deduplicate exact, near-exact, and paragraph-level duplicates.
6. Decontaminate against evaluation sets: IndoNLU, IndoMMLU-style sets, NusaX, custom eval prompts.
7. Pack into train shards with source metadata.

Recommended first run:

```bash
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert --dry-run
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert
```

Recommended full corpus pull:

```bash
python download_sources.py --sources core --dry-run
python download_sources.py --sources core
```

Optional CulturaX pull after accepting Hugging Face access terms:

```bash
HF_TOKEN=hf_xxx python download_sources.py --sources culturax_id
```

---

## 4. Layer 2 — Instruction Layer

Purpose: make the adapted model useful as an assistant and enterprise workflow model.

Instruction sources:

| Source | Use |
|---|---|
| Synthetic Indonesian instructions | general assistant behavior |
| Localized reasoning tasks | math, business reasoning, process analysis in Bahasa Indonesia |
| IndoNLU train/dev conversions | small supervised seed, not final benchmark test leakage |
| Enterprise workflow tasks | extraction, summarization, classification, table QA, tool-call JSON |
| Multimodal document tasks | scanned-page QA, OCR correction, chart/table extraction |

Example SFT record:

```json
{
  "instruction": "Jelaskan konsep stok opname untuk tim operasional retail.",
  "input": "",
  "output": "Stok opname adalah proses menghitung dan mencocokkan persediaan fisik dengan catatan sistem..."
}
```

Rules:

- Never train on benchmark test answers.
- Keep source/provenance metadata.
- Separate public, synthetic, and permissioned data.
- Keep safety/refusal examples in Indonesian.
- Use schema-constrained examples for tool calls.

---

## 5. Layer 3 — Evaluation Layer

Purpose: prove the model improved in Indonesian without contaminating the training set.

Core evaluation assets:

| Asset | Role |
|---|---|
| IndoNLU | benchmark for Indonesian NLU tasks |
| IndoBERT | baseline/reference encoder model |
| NusaX sentiment | local-language and multilingual sentiment evaluation |
| Upstream Nemotron | before/after comparison |
| Custom enterprise eval | extraction, summarization, table QA, grounded answers, tool JSON |

Comparison matrix:

```text
IndoBERT baseline
vs upstream Nemotron 3 Nano Omni
vs Nemotron-Indonesia CPT
vs Nemotron-Indonesia SFT
vs Nemotron-Indonesia final/aligned
```

Success criteria:

- Improved Indonesian comprehension vs upstream Nemotron.
- Competitive NLU score vs IndoBERT-style baselines where task-compatible.
- Stronger Bahasa generation and enterprise-task reliability.
- No benchmark contamination.
- Local deployability on NVIDIA stack.

---

## 6. Engineering Flow

```text
1. Pull sources using download_sources.py
2. Build raw manifest with hashes and licenses
3. Clean + normalize + language-filter
4. Deduplicate
5. Decontaminate against eval sets
6. Pack CPT shards
7. Run small CPT smoke test
8. Evaluate upstream vs CPT checkpoint
9. Build SFT set
10. Run SFT / multimodal SFT
11. Evaluate again
12. Prepare release candidate + inference profile
```

---

## 7. Why IndoBERT and IndoNLU Matter

IndoBERT is not the base model for this project. It is an encoder-style baseline and helper.

Use IndoBERT for:

- benchmark comparison;
- text-quality filtering experiments;
- NER/classification helper baselines;
- credibility: measuring against established Indonesian NLP work.

IndoNLU is not a large training corpus. It is primarily the benchmark/evaluation suite, with limited train/dev data usable as supervised seed if test leakage is avoided.

Use IndoNLU for:

- official Indonesian NLU evaluation;
- controlled SFT conversions from non-test splits;
- decontamination target for corpus filtering;
- scorecard reporting.

---

## 8. Minimal First Milestone

The first useful milestone is not a full 20B-token run. It is a small, reproducible proof:

```text
Data: Indo4B HF mirror + Wikipedia ID + IndoNLU eval + IndoBERT baseline
Run: small CPT smoke test
Eval: upstream Nemotron vs CPT checkpoint on IndoNLU/custom Bahasa prompts
Output: scorecard + sample generations + next-run recommendation
```

That is enough to validate the pipeline before burning full H200 time.
