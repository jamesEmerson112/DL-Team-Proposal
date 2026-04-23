# Parameter Golf — Submission Guide

> Step-by-step guide for submitting to OpenAI's Parameter Golf competition.
> For setup and background, see [00_overview.md](00_overview.md).
> For experiment results, see [findings.md](findings.md).

---

## Pre-Submission Checklist

Before preparing your submission, confirm all of the following:

- [ ] **Score**: val_bpb beats baseline (1.2244) OR you have a unique non-record approach
- [ ] **Budget**: artifact size <= 16,000,000 bytes (code + int8+zlib compressed model)
- [ ] **Hardware**: ran on 8xH100 SXM within 10-min wall clock (600 seconds)
- [ ] **Code**: `train_gpt.py` is self-contained, compiles and runs from records folder (1500-line limit is for main repo script only, not records/)
- [ ] **No cheating**: no external downloads during eval, no training on validation data
- [ ] **Seeds**: 3 seeds recommended (modern standard); required for SOTA claims (p < 0.01)
- [ ] **Compliance**: all 9 compliance flags verified (see compliance section below)
- [ ] **SOTA only**: if claiming a new record, beat current SOTA by >= 0.005 nats at p < 0.01

---

## Required Files

Every submission needs exactly 4 items in one folder. No more, no less.

### 1. `submission.json`

**Modern format (used by rank 1-5 submissions, April 2026):**

```json
{
  "author": "Your Name",
  "github_id": "your-github-username",
  "name": "Short Descriptive Title",
  "blurb": "1-2 sentence summary. Include key techniques and final val_bpb.",
  "date": "2026-04-XX",
  "track": "10min_16mb",
  "val_bpb": 1.XXXX,
  "val_bpb_std": 0.XXXXX,
  "seeds": [42, 1337, 2025],
  "seed_results": {
    "42":   {"val_bpb": 1.XXXX, "val_loss": 3.XXXX, "artifact_bytes": 15XXXXXX, "train_time_ms": 600000, "ttt_time_ms": 160000},
    "1337": {"val_bpb": 1.XXXX, "val_loss": 3.XXXX, "artifact_bytes": 15XXXXXX, "train_time_ms": 600000, "ttt_time_ms": 160000},
    "2025": {"val_bpb": 1.XXXX, "val_loss": 3.XXXX, "artifact_bytes": 15XXXXXX, "train_time_ms": 600000, "ttt_time_ms": 160000}
  },
  "bytes_total": 15XXXXXX,
  "bytes_code": XXXXX,
  "hardware": "8xH100 80GB SXM",
  "pytorch_version": "2.X.X",
  "tokenizer": "SentencePiece BPE 8192",
  "architecture": "9L 448d 8h 4kv GQA headwise-gate LeakyReLU2 QK5.0 tied-embed",
  "compliance": {
    "train_under_600s": true,
    "artifact_under_16mb": true,
    "eval_under_600s": true,
    "no_slot": true,
    "no_pre_quant_ttt": true,
    "no_etlb": true,
    "no_ngram_cache": true,
    "score_first_ttt": true,
    "three_seeds": true
  },
  "attribution": {
    "SP8192": "@kevclark (PR #1394)",
    "LeakyReLU2": "@abaybektursun (PR #549)",
    "Score-First TTT": "@dexhunter (PR #1413)",
    "QK-Gain 5.0": "@dexhunter (PR #1413)"
  }
}
```

**Field notes:**
- `val_bpb` — mean across seeds. Use `final_int8_ttt_exact` value if TTT enabled, otherwise `final_int8_zlib_roundtrip_exact`.
- `val_bpb_std` — standard deviation across seeds. Shows training stability.
- `seed_results` — per-seed breakdown. Modern submissions track artifact_bytes per seed.
- `bytes_total` — compressed model (.ptz) + train_gpt.py size in bytes. Must be <= 16,000,000.
- `bytes_code` — just the `train_gpt.py` file size in bytes (`wc -c < train_gpt.py`).
- `compliance` — 9 boolean flags (see Compliance section below). All modern submissions include this.
- `attribution` — credit each technique with `@github_user (PR #N)` format. Required.
- `date` — ISO 8601 format. Use the date you submit the PR.

**Minimal format (still accepted for non-SOTA submissions):**
```json
{
  "author": "Your Name",
  "github_id": "your-github-username",
  "name": "Short Descriptive Title",
  "blurb": "Summary with key techniques and val_bpb.",
  "date": "2026-04-XX",
  "val_loss": 3.XXXXX,
  "val_bpb": 1.XXXX,
  "bytes_total": 15XXXXXX,
  "bytes_code": XXXXX
}
```

### 2. `README.md`

Structure your README like top submissions do:

```markdown
# Title of Your Submission

**val_bpb: X.XXXX** (N-seed mean, std X.XXXX) | **~XX.XX MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch X.XX)

| Seed | step_avg | steps | val_bpb | Artifact Size |
|------|----------|-------|---------|---------------|
| 42   | XXms     | X,XXX | X.XXXX  | XX,XXX,XXX    |
| 1337 | XXms     | X,XXX | X.XXXX  | XX,XXX,XXX    |
| 2025 | XXms     | X,XXX | X.XXXX  | XX,XXX,XXX    |
| Mean | XXms     | X,XXX | X.XXXX (std X.XXXX) |  |

## Key Techniques

List each technique with PR attribution:
- **SP8192** — 8192-token SentencePiece BPE (@kevclark, PR #1394)
- **Score-First TTT** — legal test-time training (@dexhunter, PR #1413)
- etc.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers    | ...     |
| Dims      | ...     |
| Heads     | ...     |
| KV Heads  | ...     |
| Activation | ...   |
| Vocab     | ...     |
| Quantization | int8+zlib |

## Compliance

- [x] Training under 600s
- [x] Artifact under 16 MB
- [x] Eval under 600s (including TTT)
- [x] No SLOT (supervised learning on test)
- [x] No pre-quantization TTT
- [x] No ETLB (eval-time learned biases)
- [x] No n-gram cache
- [x] Score-first TTT (score before update)
- [x] Three seeds

## Run Command

Full torchrun command with all env vars.

## Ablation (optional but encouraged)

Incremental contribution of each technique.

## Credits

Link to any PRs, papers, or prior submissions you built on.
Use format: @github_username (PR #XXXX)

## Included Files

- README.md (this file)
- submission.json
- train_gpt.py
- train_seed42.log, train_seed1337.log, train_seed2025.log
```

### 3. Training Log(s)

- **Single seed**: name it `train.log`
- **Multiple seeds**: name them `train_seed42.log`, `train_seed1337.log`, `train_seed2025.log`
- This is the raw stdout from your training run — do not edit or truncate it
- Must show `final_int8_zlib_roundtrip_exact val_loss:X val_bpb:Y` at the end

### 4. `train_gpt.py`

- Exact snapshot of the training script you used
- Must be self-contained — no imports from outside the repo's installed packages
- Must compile and run from the records folder (test this!)
- The 1500-line limit applies to PRs on the main repo script, NOT to submissions in records/
- Code size counts toward the 16 MB budget (code bytes + model bytes <= 16,000,000)

---

## Folder Naming

**Format:** `YYYY-MM-DD_DescriptiveTitle`

**Location:**
- Record submissions: `records/track_10min_16mb/`
- Non-record submissions: `records/track_non_record_16mb/`

**Examples from real submissions:**
```
2026-03-17_NaiveBaseline
2026-03-23_LeakyReLU_LegalTTT_ParallelMuon
2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2
2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
```

**Tips:**
- Use underscores, not spaces
- Lead with the most important technique
- Include vocab variant if not SP1024 (e.g., SP8192, SP4096)
- Keep it scannable — someone should guess what you did from the folder name

---

## Step-by-Step PR Workflow

### Step 1: Sync your fork

```bash
cd /path/to/parameter-golf
git fetch upstream
git merge upstream/main
git push origin main
```

If you don't have the upstream remote set:
```bash
git remote add upstream https://github.com/openai/parameter-golf.git
```

### Step 2: Create a branch

```bash
git checkout -b submission/YYYY-MM-DD_YourTitle
```

### Step 3: Create your submission folder

```bash
mkdir -p records/track_10min_16mb/YYYY-MM-DD_YourTitle
```

### Step 4: Copy your files in

```bash
# Copy your modified training script
cp train_gpt.py records/track_10min_16mb/YYYY-MM-DD_YourTitle/

# Copy training log(s) — from your RunPod pod or local logs/ directory
cp logs/your_run.txt records/track_10min_16mb/YYYY-MM-DD_YourTitle/train.log

# Create submission.json and README.md (see templates above)
```

### Step 5: Verify train_gpt.py works standalone

```bash
cd records/track_10min_16mb/YYYY-MM-DD_YourTitle/
python3 -c "import train_gpt"   # should not error
cd ../../..
```

### Step 6: Verify artifact size

```bash
# Code size
wc -c < records/track_10min_16mb/YYYY-MM-DD_YourTitle/train_gpt.py

# Compressed model size (from your training log, look for "int8+zlib" size)
# OR if you have the .ptz file:
wc -c < /path/to/your_model.ptz

# Total must be <= 16,000,000 bytes
```

### Step 7: Commit and push

```bash
git add records/track_10min_16mb/YYYY-MM-DD_YourTitle/
git commit -m "Add submission: YourTitle (val_bpb X.XXXX)"
git push -u origin submission/YYYY-MM-DD_YourTitle
```

### Step 8: Create the PR

```bash
gh pr create \
  --repo openai/parameter-golf \
  --title "Record: YourTitle (val_bpb X.XXXX)" \
  --body "## Summary
- val_bpb: X.XXXX (N-seed mean)
- Artifact: XX.XX MB (under 16 MB)
- Techniques: list your key techniques

See records/track_10min_16mb/YYYY-MM-DD_YourTitle/README.md for full details."
```

For non-record submissions, use `"Non-record: ..."` in the title.

### Step 9: Wait for review

OpenAI reviews submissions chronologically by PR creation time. They check:
- Reproducibility (log evidence matches claimed score)
- Statistical significance (3+ seeds for SOTA claims)
- Code correctness (train_gpt.py compiles and runs)
- Spirit of the challenge (no gaming)

---

## Team-Specific Workflow

Our repo has tooling that automates most of the data collection.

### Using the run script summary

After `runs/parameter_golf_baseline.sh` finishes, it prints a summary block like:

```
run_id:       your_run_name
gpus:         8
params:       17,020,480
val_loss_raw: 2.1338
val_bpb_raw:  1.2638
val_loss_int8:2.1405
val_bpb_int8: 1.2645
size_int8:    14.65 MB
under_budget: YES
baseline:     1.2244
gap:          +0.0401
```

Map this to submission.json:
- `val_loss` = `val_loss_int8` (or `val_loss_ttt` if TTT was used)
- `val_bpb` = `val_bpb_int8` (or `val_bpb_ttt` if TTT was used)
- `bytes_total` = compressed artifact bytes + train_gpt.py bytes

### Where to find artifacts

- **Training logs**: `parameter-golf/logs/${RUN_ID}.txt`
- **Compressed model**: look for `*.ptz` files in the parameter-golf directory after training
- **train_gpt.py**: `parameter-golf/train_gpt.py` (copy at time of run)

### Running multiple seeds for statistical significance

For SOTA claims, run 3 seeds. Change only the `SEED` env var:

```bash
# Seed 1
source runs/configs/your_config.env
SEED=42 bash runs/parameter_golf_baseline.sh
cp parameter-golf/logs/${RUN_ID}.txt submission_folder/train_seed42.log

# Seed 2
SEED=1337 bash runs/parameter_golf_baseline.sh
cp parameter-golf/logs/${RUN_ID}.txt submission_folder/train_seed1337.log

# Seed 3
SEED=2025 bash runs/parameter_golf_baseline.sh
cp parameter-golf/logs/${RUN_ID}.txt submission_folder/train_seed2025.log
```

Use the **mean val_bpb** across seeds for submission.json. Include per-seed results in your README.

### Our fork

Our fork: `github.com/jamesEmerson112/parameter-golf`

```bash
# First-time setup (already done)
git clone https://github.com/jamesEmerson112/parameter-golf.git
cd parameter-golf
git remote add upstream https://github.com/openai/parameter-golf.git
```

---

## Common Pitfalls

| Pitfall | What goes wrong | How to avoid |
|---------|-----------------|--------------|
| **16 MB vs 16 MiB** | You think 16 MB = 16,777,216 bytes. It's actually 16,000,000 (decimal). | Always check `bytes_total <= 16000000` |
| **Stale env vars** | Previous `source config.env` leaves variables in your shell. Your "baseline" run uses gated attention from a previous config. | Every config must set ALL toggles. Start a fresh shell between runs. |
| **train_gpt.py doesn't run standalone** | Your script imports a local helper file or assumes a specific working directory. | Test: `cd records/.../your_folder && python3 -c "import train_gpt"` |
| **Eval budget forgotten** | Training takes 10 min, but eval (especially with TTT) also has a 10-min cap. Total can be up to 20 min. | TTT adds 400-500s. Budget accordingly. |
| **Training on val data** | TTT on validation tokens you haven't scored yet. | Score-first protocol only: adapt on chunks you've already evaluated. |
| **Single seed for SOTA** | Your improvement is within noise. Rejected. | 3+ seeds, report mean and std, show p < 0.01. |
| **Edited training log** | Truncated or modified log. Reviewers notice. | Submit raw stdout, never edit. |
| **Wrong val_bpb in submission.json** | You use the pre-compression value instead of post-int8+zlib. | Use `final_int8_zlib_roundtrip_exact` (or `final_int8_ttt_exact` with TTT). |

---

## Non-Record Submissions

Don't need to beat SOTA? You can still submit to the **unlimited compute track**.

**Same 4 files required**, placed in `records/track_non_record_16mb/`.

**Lower bar:**
- No statistical significance requirement
- No 10-min training time constraint
- Must still fit in 16 MB artifact
- Must still run and produce valid val_bpb

**Good candidates:**
- Creative/weird architectures (ternary quantization, state-space models, text diffusion)
- In-progress work that shows signs of life
- Interesting negative results with good analysis
- Techniques that need more compute to mature

Note in your README that this is a non-record submission and explain why it's interesting.

---

## Quick Reference

| Item | Value |
|------|-------|
| Deadline | April 30, 2026 |
| Baseline BPB | 1.2244 |
| Current SOTA | ~1.0810 |
| Artifact cap | 16,000,000 bytes (decimal) |
| Training time | 10 min on 8xH100 SXM |
| Eval time | 10 min (separate budget) |
| Code line limit | 1500 lines |
| SOTA threshold | >= 0.005 nats improvement, p < 0.01 |
| Our fork | github.com/jamesEmerson112/parameter-golf |
| Discord | #parameter-golf-discussions |
| Upstream repo | github.com/openai/parameter-golf |
