# Operator Runbook

This is the fastest path for the training operator to set up the machine, pull the first data sources, and run the first validation pass.

The first goal is not a full training run. The first goal is to prove the pipeline works end to end:

```text
clone repo
→ install environment
→ verify GPU + tokenizer
→ verify source links
→ pull small first-milestone data
→ run baseline evaluation
→ prepare first corpus shard
→ start small CPT smoke test
→ report results
```

---

## 0. Machine Assumptions

Recommended:

- 8× NVIDIA H200 for full adaptation.
- H100 80GB is acceptable for smaller experiments.
- Ubuntu/Linux with CUDA drivers already installed.
- Conda or Mamba installed.
- Enough disk for data. Start with at least 500GB free for the first milestone. Full corpus pulls can need multiple TB.

Quick checks:

```bash
nvidia-smi
df -h
python3 --version
```

If `nvidia-smi` fails, stop and fix the GPU/driver setup first.

---

## 1. Clone the Repo

```bash
git clone https://github.com/lucacadalora/nemotron-indonesia.git
cd nemotron-indonesia
```

---

## 2. Run Setup Check

This installs Python dependencies, checks the tokenizer, and prints the next commands. It does **not** download the large datasets.

```bash
bash START_HERE.sh
```

If the setup script fails, send the last 50 lines of the terminal output before continuing.

---

## 3. Verify Data Links Before Downloading

```bash
python download_sources.py --sources first_milestone --verify-links --dry-run
```

Expected: every line should return `OK 200`.

The first-milestone bundle is:

| Source | Why it is included |
|---|---|
| `indo4b_hf` | main Indonesian corpus seed |
| `wikipedia_id` | clean structured Indonesian knowledge |
| `indonlu` | benchmark/evaluation suite |
| `indobert` | Indonesian baseline model |

If a link fails, do not start a full pull. Fix the source first.

---

## 4. Pull First-Milestone Data

```bash
python download_sources.py --sources first_milestone
```

Output locations:

```text
data/raw/indo4b-hf/
data/raw/wikipedia-id/
data/eval/indonlu/
models/baselines/indobert-base-p1/
```

Check disk after the pull:

```bash
du -sh data models
df -h
```

---

## 5. Optional: Pull the Full Core Bundle

Only do this after the first milestone works.

```bash
python download_sources.py --sources core --verify-links --dry-run
python download_sources.py --sources core
```

Core bundle:

```text
indo4b_hf
sea_pile_id
mc4_id
cc100_id
wikipedia_id
indonlu
nusax_senti
indobert
```

Optional CulturaX requires accepting Hugging Face access terms first:

```bash
HF_TOKEN=hf_xxx python download_sources.py --sources culturax_id
```

---

## 6. Validate Base Model Tokenizer

```bash
python - <<'PY'
from transformers import AutoTokenizer
model = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("tokenizer ok")
print("vocab size:", len(tok))
PY
```

Expected: `tokenizer ok` plus a vocab size.

---

## 7. Run Baseline Evaluation First

Before training, run the upstream model/baseline eval so we have a comparison point.

```bash
mkdir -p results
python evaluate.py \
  --model_path nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --benchmark indonlu \
  --output results/upstream_indonlu.json
```

If full model loading is too heavy at this stage, record the failure and continue with tokenizer/data validation. Do not hide the failure; it is useful for sizing the run.

---

## 8. Prepare First Text Corpus

`prepare_data.py` reads from whatever was already downloaded by `download_sources.py`.
Always pass `--data_dir ./data/raw` so the script uses local files instead of re-downloading.

**First-milestone (indo4b_hf + wikipedia only):**

```bash
python prepare_data.py \
  --data_dir ./data/raw \
  --output_dir ./data/processed \
  --datasets indo4b_hf wikipedia \
  --tokenizer nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --dedup_threshold 0.85
```

**Full core bundle (after step 5 completes):**

```bash
python prepare_data.py \
  --data_dir ./data/raw \
  --output_dir ./data/processed \
  --datasets indo4b_hf wikipedia seapile cc100 mc4_id \
  --tokenizer nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --dedup_threshold 0.85
```

**With CulturaX (if pulled in step 5):**

```bash
python prepare_data.py \
  --data_dir ./data/raw \
  --output_dir ./data/processed \
  --datasets all \
  --tokenizer nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --dedup_threshold 0.85
```

Available `--datasets` values: `indo4b_hf`, `wikipedia`, `seapile`, `cc100`, `mc4_id`, `culturax_id`, `liputan6`, `kaskus`, `all`

Expected output:

```text
data/processed/indonesian_corpus/
data/processed/tokenizer/
```

Check:

```bash
du -sh data/processed
```

---

## 9. Start Small CPT Smoke Test

Do not start the full multi-day run until data prep and baseline evaluation are confirmed.

```bash
./run_training.sh pretrain
```

For the first smoke test, watch:

```bash
nvidia-smi
ls -lah models/
```

Stop early if:

- loss is `nan`;
- GPU memory is unstable;
- checkpoint writes fail;
- data loader hangs;
- disk usage is close to full.

---

## 10. What to Send Back After the First Run

Send a short report with:

```text
1. Machine: GPU type/count, CUDA version, free disk
2. Sources pulled: source names + data size
3. Tokenizer check: pass/fail
4. Baseline eval: output file or error
5. Data prep: output path + number of examples if available
6. Training smoke test: steps completed, loss sample, checkpoint path
7. Blockers/questions
```

Example:

```text
Machine: 8x H200, CUDA 12.x, 3.2TB free
Pulled: indo4b_hf, wikipedia_id, indonlu, indobert
Tokenizer: OK
Baseline eval: failed loading full BF16 model, likely memory/config issue; log attached
Data prep: data/processed/indonesian_corpus created
Training: not started yet
Blocker: need confirm NeMo/Megatron path for full Omni training
```

---

## 11. Common Issues

### Hugging Face access error

For public sources, retry login:

```bash
huggingface-cli login
```

For CulturaX, accept dataset terms on Hugging Face first, then run with `HF_TOKEN`.

### Disk fills up

Stop downloads/training, then check:

```bash
du -h --max-depth=2 data models | sort -h | tail -30
df -h
```

Do not delete checkpoints unless the training owner confirms.

### Flash Attention install fails

The setup script continues without it. Record the failure. It may reduce speed but should not block early validation.

### Full model loading fails

Record GPU memory, command, and traceback. For full Omni training, switch to NVIDIA NeMo/Megatron-Bridge recipes rather than forcing the simple Hugging Face scaffold.

---

## 12. Files to Read in Order

1. `OPERATOR_RUNBOOK.md` — this file.
2. `ARCHITECTURE.md` — model/data/eval architecture.
3. `DATASET_MANIFEST.md` — source links and download commands.
4. `README.md` — general repo overview.
5. `NEMOTRON_3_NANO_OMNI_GITHUB_REVIEW.md` — NVIDIA source review.

Start here, validate small, then scale.
