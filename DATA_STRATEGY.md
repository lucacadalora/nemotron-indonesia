# Nemotron-Indonesia 30B — RIGOROUS DATA STRATEGY
## For Maximum IndoMMLU Benchmark Performance

**Version:** 1.1  
**Date:** April 24, 2026  
**Status:** Production-Ready Pipeline  
**Goal:** Beat Sahabat AI 70B (52%) → Target 55%+ on IndoMMLU

---

## 1. Philosophy: Quality >> Quantity

Sahabat AI used 50B tokens and got 52% on IndoMMLU. More data ≠ better. We need **curated, decontaminated, education-heavy** data.

| Approach | Sahabat AI | Nemotron-Indonesia |
|----------|-----------|-------------------|
| Token count | 50B (quantity) | **20B (quality)** |
| Benchmark decontamination | Unknown | **Aggressive 13-gram removal** |
| Educational content | ~2% (Wiki/News) | **25%+ (textbooks, exams, academic)** |
| Curriculum learning | No | **Yes — easy → hard** |
| Quality filtering | Basic | **Multi-signal (perplexity, length, lang-id, toxicity)** |
| Data mixing | Single epoch | **Optimized ratios by domain** |

---

## 2. Pre-Training Corpus: 20B Tokens (Rigorous Mix)

### 2.1 Data Sources & Ratios

| Source | Tokens | % | Quality Tier | Purpose for IndoMMLU |
|--------|--------|---|--------------|---------------------|
| **Indonesian Wikipedia** | 2.0B | 10% | A+ | Structured knowledge — STEM, history, geography |
| **Indonesian Academic Corpus** | 2.5B | 12.5% | A+ | Scientific papers, textbooks, educational content |
| **OSCAR (filtered)** | 5.0B | 25% | B+ | Broad web coverage, deduplicated & quality-scored |
| **CC100 Indonesian (filtered)** | 3.0B | 15% | B+ | Web diversity, filtered for quality |
| **SEA-LION Pile - Indonesian** | 1.0B | 5% | A | AI Singapore-curated SEA language corpus |
| **Liputan6 + ID News** | 2.0B | 10% | A | Formal language, current events, social science |
| **Kaskus (heavily filtered)** | 1.0B | 5% | C+ | Informal Indonesian, internet slang, cultural context |
| **Indonesian Government Docs** | 1.5B | 7.5% | A | Legal, policy, civic education (civics exam content) |
| **Religious Texts (Quran, Bible ID)** | 0.5B | 2.5% | A | Cultural/religious literacy (common exam topics) |
| **Javanese/Sundanese/Balinese** | 1.5B | 7.5% | B | Local language coverage for NusaX + cultural questions |
| **English Academic (STEM)** | 1.0B | 5% | A+ | STEM concepts transfer to Indonesian (math, physics) |
| **TOTAL** | **20.0B** | **100%** | | |

### 2.2 Quality Tiers Explained

**A+ (Textbook-grade):**
- Indonesian Wikipedia ( dumps.wikimedia.org )
- Academic papers from Indonesia (Garuda, SINTA-indexed)
- Indonesian curriculum textbooks (BSE — Buku Sekolah Elektronik from Kemdikbud)
- Scientific abstracts in Indonesian

**A (News-grade formal):**
- Liputan6 (215K articles → expand to full corpus)
- Kompas, Tempo, Republika, Antara news archives
- VOA Indonesian, Global Voices

**B+ (Filtered web):**
- OSCAR 2301 Indonesian subset → filter with quality heuristics
- CC100 Indonesian → deduplicate + quality score

**C+ (Informal, heavily filtered):**
- Kaskus forums → only posts >200 chars, language confidence >0.95, toxicity score <0.3

---

### 2.3 SEA-LION Pile — AI Singapore Collaboration Asset

SEA-LION (Southeast Asian Languages in One Network) is AI Singapore's flagship multilingual LLM project. Their **SEA-LION Pile** dataset is openly available and represents the highest-quality curated Indonesian web corpus.

**Why we include it:**
- ✅ Already used by Sahabat AI (27.5B tokens, 55% of their pre-training data)
- ✅ AI Singapore is a **proven collaborator** — they co-developed Sahabat AI with GoTo
- ✅ Opens door to future partnership with AI Singapore (NUS-backed, government-funded)
- ✅ Higher quality than raw OSCAR/CC100 — already filtered for SEA languages
- ✅ Includes Javanese, Sundanese, Balinese subsets for local language coverage

**Dataset:** https://huggingface.co/datasets/aisingapore/sea-lion-pile

**Note:** Use SEA-LION Pile v2 (latest) which filters CommonCrawl WARC with fastText language classifier for 11 SEA languages.

---

## 3. RIGOROUS Processing Pipeline

### Stage 1: Download & Initial Filter
```python
# For each source:
1. Download raw data
2. Remove HTML tags, URLs, email addresses, phone numbers
3. Remove lines with <50% Indonesian characters (unicode range)
4. Remove documents <100 chars or >100K chars
5. Language detection: fastText Indonesian confidence > 0.90
```

### Stage 2: Quality Scoring (Multi-Signal)

Every document gets a quality score (0-1) based on:

| Signal | Weight | How |
|--------|--------|-----|
| **Perplexity** | 30% | Score against a small Indonesian GPT-2. Lower = better. Reject if perplexity > 500 |
| **Language confidence** | 20% | fastText Indonesian score. Must be >0.95 |
| **Length** | 15% | Prefer 200-2000 char documents. Penalize extremes |
| **Toxicity** | 15% | Reject if toxicity classifier >0.5 |
| **Repetition** | 10% | Reject if >30% of lines are duplicates |
| **Readability** | 10% | Flesch-Kincaid adapted for Indonesian. Prefer mid-range |

**Threshold:** Keep only documents with composite quality score > 0.65

### Stage 3: Deduplication (3 Levels)

```python
# Level 1: Exact dedup (MD5 hash of normalized text)
# Level 2: Near-exact (MinHash LSH, Jaccard > 0.95)
# Level 3: Fuzzy paragraph (simhash, 13-gram overlap > 0.85)
```

**Expected dedup rate:** 40-60% of raw web data removed

### Stage 4: BENCHMARK DECONTAMINATION (CRITICAL)

This is where most teams fail. We must remove ANY overlap with test sets.

```python
# Decontaminate against:
1. IndoMMLU test set (all 14K questions + answers)
2. IndoMMLU validation set
3. NusaX test set
4. IndoNLI test set
5. SEA-HELM / BHASA test prompts
6. IndoSum test set

# Method:
- Extract all n-grams (n=5,8,13) from test sets
- Build Bloom filter for O(1) lookup
- For each training document:
  - If ANY 13-gram matches test set → REJECT document
  - If >5% of 8-grams match → REJECT document
  - If >2% of 5-grams match → FLAG for manual review

# Expected removal: 0.1-2% of corpus (but critical for validity)
```

### Stage 5: Curriculum Learning Sorting

Sort remaining documents by "difficulty":

```python
difficulty_score = (
    0.4 * perplexity_score +      # Harder = higher perplexity
    0.3 * domain_complexity +      # STEM > news > casual
    0.2 * vocabulary_rarity +      # Rare words = harder
    0.1 * sentence_length          # Longer = harder
)

# Shuffle within difficulty buckets
# Train: easy → medium → hard (first 30% easy, 40% medium, 30% hard)
```

### Stage 6: Tokenization & Language Tags

```python
# Tokenize with Nemotron tokenizer
# Add language tokens for multilingual segments:
<|id|>    → Indonesian
<|jv|>    → Javanese  
<|su|>    → Sundanese
<|ban|>   → Balinese
<|min|>   → Minangkabau
<|en|>    → English (for STEM transfer)

# Format:
<|id|> {indonesian text}
<|jv|> {javanese text}
```

---

## 4. SFT Data (500K Pairs): Benchmark-Optimized

### 4.1 Instruction Sources

| Source | Pairs | Type | Quality |
|--------|-------|------|---------|
| **IndoMMLU training split** | 50K | Academic Q&A (NOT test!) | A+ |
| **Synthetic textbooks** | 100K | Generated from BSE textbooks | A+ |
| **NusaX sentiment + NLI** | 30K | Classification tasks | A |
| **Translated Flan/CoT** | 150K | English reasoning tasks → ID | A |
| **Government FAQ** | 50K | Civic, legal, administrative | A |
| **Custom agentic instructions** | 120K | Tool use, multi-step, coding | A |

### 4.2 IndoMMLU Training Split (CRITICAL)

IndoMMLU has train/val/test splits. **Only use train split for SFT.**

```python
# IndoMMLU covers:
# - Humanities (history, literature, religion)
# - Indonesian language (grammar, vocabulary)
# - Local languages & cultures
# - Social science (economics, geography, civics)
# - STEM (math, physics, chemistry, biology)
# - Primary, middle, high school levels

# Format as instruction-following:
{
  "instruction": "Berikut adalah soal ujian untuk siswa SMP. Pilih jawaban yang benar.\n\nSoal: {question}",
  "input": "",
  "output": "{answer}\n\nPenjelasan: {explanation}"
}
```

### 4.3 Synthetic Textbook Data (Secret Weapon)

Generate 100K instruction pairs from Indonesian curriculum:

```python
# Source: BSE (Buku Sekolah Elektronik) from Kemdikbud
# Download: https://bse.kemdikbud.go.id/

# For each textbook chapter:
1. Extract key concepts
2. Generate 5 question types:
   - Factual recall
   - Concept explanation
   - Application problem
   - Comparison/analysis
   - Evaluation/synthesis

3. Generate with small LLM (Qwen 7B), then filter:
   - Must be answerable from text
   - Must have unambiguous correct answer
   - Must be appropriate difficulty level
```

---

## 5. DPO Data (50K Pairs): Preference Optimization

### 5.1 Pair Generation

For each prompt, generate 4 responses using the SFT model:
- Score each with a reward model (or GPT-4 as judge)
- Keep best = "chosen", worst = "rejected"

### 5.2 Prompt Categories

| Category | % | Purpose |
|----------|---|---------|
| IndoMMLU-style questions | 30% | Direct benchmark alignment |
| Helpfulness (general) | 25% | General assistant quality |
| Safety/harmlessness | 20% | Refuse harmful requests |
| Local cultural sensitivity | 15% | Respect Indonesian values |
| Agentic/tool use | 10% | Function calling accuracy |

---

## 6. Quality Validation Checkpoints

### 6.1 Before Pre-Training
- [ ] All sources downloaded and verified
- [ ] Quality scores computed and histogram-checked
- [ ] Deduplication stats: raw → deduped token count
- [ ] Decontamination: % of corpus removed
- [ ] Spot-check 1000 random documents (human review)
- [ ] Language distribution verified ( Indonesian >75% )

### 6.2 During Pre-Training (Every 500 steps)
- [ ] IndoMMLU validation score (NOT test — save test for final)
- [ ] Perplexity on held-out validation set
- [ ] Training loss curve (watch for spikes)
- [ ] Gradient norm check (watch for instability)

### 6.3 Final Evaluation Protocol
```python
# STRICT PROTOCOL:
1. Run IndoMMLU test set ONCE (never during training/tuning)
2. 5-shot with Indonesian prompts
3. Temperature = 0 (greedy decoding)
4. Report confidence intervals (bootstrap over questions)
5. Compare against Sahabat AI 70B with SAME evaluation code
```

---

## 7. Expected IndoMMLU Breakdown by Domain

| Domain | Sahabat AI 70B | SEA-LION v3-9B | Nemotron-ID 30B Target | Strategy |
|--------|---------------|----------------|----------------------|----------|
| **Humanities** | ~50% | ~52% | **55%+** | Heavy Wikipedia + news + religious texts |
| **Indonesian Language** | ~55% | ~58% | **60%+** | Formal corpus + grammar-focused SFT |
| **Local Languages** | ~45% | ~60% | **55%+** | Javanese/Sundanese/Balinese pre-train data |
| **Social Science** | ~52% | ~54% | **58%+** | Government docs + news + civics textbooks |
| **STEM** | ~48% | ~50% | **55%+** | Academic corpus + English STEM transfer |
| **Overall** | **~52%** | **~55%** | **55%+** | **Domain-balanced, education-heavy** |

---

## 8. Tools & Scripts

### 8.1 Data Processing Pipeline
```bash
# 1. Download all sources
python scripts/download_all.py --output ./data/raw/

# 2. Run quality pipeline
python scripts/process_data.py \
  --input ./data/raw/ \
  --output ./data/processed/ \
  --quality-threshold 0.65 \
  --dedup \
  --decontaminate ./data/benchmark_tests/

# 3. Verify
python scripts/verify_corpus.py ./data/processed/
```

### 8.2 Key Dependencies
```
datasets>=3.2.0
datasketch>=1.6.0      # MinHash dedup
fasttext>=0.9.2        # Language detection
transformers>=4.46.2
langdetect>=1.0.9
textstat>=0.7.3        # Readability scores
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Benchmark contamination | Aggressive 13-gram decontamination + human spot-check |
| Poor-quality web data | Multi-signal filtering, perplexity threshold |
| Overfitting to IndoMMLU | Use ONLY train split for SFT; diverse pre-training corpus |
| Data bias (urban/Jakarta-centric) | Include regional news, local languages, rural gov docs |
| Religious/cultural sensitivity | Include Quran/Bible translations, traditional texts |
| English-only STEM concepts | 5% English academic data for concept transfer |

---

## 10. Success Metrics

| Metric | Target | How to Verify |
|--------|--------|--------------|
| Corpus quality score | >0.80 avg | Sample 10K docs, manual rating |
| Deduplication rate | 40-60% | Compare raw vs processed token counts |
| Decontamination rate | <5% removed | Check against all benchmark test sets |
| Language purity | >90% Indonesian | fastText confidence histogram |
| IndoMMLU (final) | **55%+** | Single evaluation run on held-out test |
| NusaX average | **80%+** | 12-language sentiment accuracy |

---

**Next Step:** Implement the processing pipeline and run verification before starting pre-training.

**Contact:** luca@jatevo.ai | hibagus (collaborator)