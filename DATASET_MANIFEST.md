# Nemotron-Indonesia Dataset Manifest

Download helper:

```bash
python download_sources.py --list
python download_sources.py --sources core --dry-run
python download_sources.py --sources core
```

This manifest avoids manually gated OSCAR as a dependency. Optional auto-gated sources are clearly marked.

---

## Core Sources

| Source key | Purpose | Public URL | Pull command |
|---|---|---|---|
| `indo4b_hf` | Core Indonesian CPT corpus; HF parquet mirror of Indo4B | https://huggingface.co/datasets/taufiqdp/Indo4B-hf | `python download_sources.py --sources indo4b_hf` |
| `sea_pile_id` | Curated SEA web corpus, Indonesian subset | https://huggingface.co/datasets/aisingapore/SEA-PILE-v1 | `python download_sources.py --sources sea_pile_id` |
| `mc4_id` | Large Indonesian mC4 web corpus | https://huggingface.co/datasets/allenai/c4 | `python download_sources.py --sources mc4_id` |
| `cc100_id` | CC100 Indonesian corpus | https://data.statmt.org/cc-100/id.txt.xz | `python download_sources.py --sources cc100_id` |
| `wikipedia_id` | Indonesian Wikipedia dump | https://huggingface.co/datasets/wikimedia/wikipedia | `python download_sources.py --sources wikipedia_id` |
| `indonlu` | Indonesian NLU benchmark/eval suite | https://huggingface.co/datasets/indonlp/indonlu | `python download_sources.py --sources indonlu` |
| `nusax_senti` | Local-language sentiment benchmark | https://huggingface.co/datasets/indonlp/NusaX-senti | `python download_sources.py --sources nusax_senti` |
| `indobert` | IndoBERT baseline model | https://huggingface.co/indobenchmark/indobert-base-p1 | `python download_sources.py --sources indobert` |

Core bundle:

```bash
python download_sources.py --sources core --dry-run
python download_sources.py --sources core
```

---

## Optional Source

| Source key | Purpose | Access | Pull command |
|---|---|---|---|
| `culturax_id` | Large deduplicated Indonesian corpus from mC4/OSCAR-derived sources | Hugging Face auto-gated; accept terms first | `HF_TOKEN=hf_xxx python download_sources.py --sources culturax_id` |

CulturaX URL: https://huggingface.co/datasets/uonlp/CulturaX

---

## Direct Download URLs Checked

These URLs are included so the training operator can pull files manually if needed.

### Indo4B HF mirror

Dataset page:

```text
https://huggingface.co/datasets/taufiqdp/Indo4B-hf
```

Example shard:

```text
https://huggingface.co/datasets/taufiqdp/Indo4B-hf/resolve/main/data/wiki-00000-of-00001.parquet
```

CLI:

```bash
huggingface-cli download taufiqdp/Indo4B-hf \
  --repo-type dataset \
  --include 'data/*.parquet' \
  --local-dir data/raw/indo4b-hf
```

### SEA-PILE Indonesian

Dataset page:

```text
https://huggingface.co/datasets/aisingapore/SEA-PILE-v1
```

Example shard:

```text
https://huggingface.co/datasets/aisingapore/SEA-PILE-v1/resolve/main/sea-pile-mc4/id/mc4-id-00000-00020.jsonl.gz
```

CLI:

```bash
huggingface-cli download aisingapore/SEA-PILE-v1 \
  --repo-type dataset \
  --include 'sea-pile-mc4/id/*.jsonl.gz' \
  --local-dir data/raw/sea-pile-id
```

### mC4 Indonesian

Dataset page:

```text
https://huggingface.co/datasets/allenai/c4
```

Example shard:

```text
https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-id.tfrecord-00000-of-01024.json.gz
```

CLI:

```bash
huggingface-cli download allenai/c4 \
  --repo-type dataset \
  --include 'multilingual/c4-id*.json.gz' \
  --local-dir data/raw/mc4-id
```

### CC100 Indonesian + local languages

Direct files:

```text
https://data.statmt.org/cc-100/id.txt.xz
https://data.statmt.org/cc-100/jv.txt.xz
https://data.statmt.org/cc-100/su.txt.xz
```

CLI:

```bash
mkdir -p data/raw/cc100
curl -L -C - --fail -o data/raw/cc100/id.txt.xz https://data.statmt.org/cc-100/id.txt.xz
curl -L -C - --fail -o data/raw/cc100/jv.txt.xz https://data.statmt.org/cc-100/jv.txt.xz
curl -L -C - --fail -o data/raw/cc100/su.txt.xz https://data.statmt.org/cc-100/su.txt.xz
```

### Indonesian Wikipedia

Dataset page:

```text
https://huggingface.co/datasets/wikimedia/wikipedia
```

Example shard:

```text
https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.id/train-00000-of-00003.parquet
```

CLI:

```bash
huggingface-cli download wikimedia/wikipedia \
  --repo-type dataset \
  --include '20231101.id/*.parquet' \
  --local-dir data/raw/wikipedia-id
```

### IndoNLU

Dataset page:

```text
https://huggingface.co/datasets/indonlp/indonlu
```

GitHub:

```text
https://github.com/IndoNLP/indonlu
```

Paper:

```text
https://aclanthology.org/2020.aacl-main.85/
```

CLI:

```bash
huggingface-cli download indonlp/indonlu \
  --repo-type dataset \
  --local-dir data/eval/indonlu
```

### NusaX Sentiment

Dataset page:

```text
https://huggingface.co/datasets/indonlp/NusaX-senti
```

CLI:

```bash
huggingface-cli download indonlp/NusaX-senti \
  --repo-type dataset \
  --local-dir data/eval/nusax-senti
```

### IndoBERT baseline

Model page:

```text
https://huggingface.co/indobenchmark/indobert-base-p1
```

CLI:

```bash
huggingface-cli download indobenchmark/indobert-base-p1 \
  --local-dir models/baselines/indobert-base-p1
```

---

## Recommended First Pull for Pipeline Validation

This keeps the first run small enough to debug:

```bash
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert --dry-run
python download_sources.py --sources indo4b_hf wikipedia_id indonlu indobert
```

Then validate:

```bash
python evaluate.py \
  --model_path nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --benchmark indonlu \
  --output results/upstream_indonlu.json
```
