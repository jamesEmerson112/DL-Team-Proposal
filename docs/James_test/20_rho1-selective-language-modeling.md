# 20 — Rho-1: Selective Language Modeling for Parameter Golf

**Author:** James Vo
**Date:** 2026-04-26
**Paper:** [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)
**Venue:** NeurIPS 2024 Best Paper Runner-Up
**Code repo:** [microsoft/rho](https://github.com/microsoft/rho) (models + eval only, no training code)

---

## 1. Paper Overview

Rho-1 introduces **Selective Language Modeling (SLM)** — instead of computing loss on every token equally during pretraining, score tokens by how informative they are and only backprop on the most useful ones.

"Rho" (ρ) stands for information **density** — the paper focuses training on tokens with the highest density of learnable signal.

### Key Results

| Result | Number |
|--------|--------|
| Math benchmark improvement over standard training | **+16%** (GSM8k, MATH) |
| Convergence speedup | **10x faster** to reach baseline accuracy |
| Data efficiency | Matches DeepSeekMath-7B with **15B vs 500B tokens** (33x less data) |
| General benchmark improvement (15 tasks) | **+6.8% average** |
| Self-referencing improvement (no curated data) | **+3.3% average** |
| Token selection ratio | **60-70%** (drop 30-40% of tokens) |

Rho-1-1B (a 1B param model) scored 40.6% on MATH — first ever 1B model to exceed 40%, nearly matching early GPT-4's chain-of-thought score of 42.5%.

---

## 2. How SLM Works

### The Core Insight: Not All Tokens Are Equal

The paper analyzed how per-token loss evolves during training and found 4 categories:

```
                        AT THE END OF TRAINING
                    +---------------+---------------+
                    |  Still Easy   |  Became Hard   |
        +-----------+---------------+---------------+
 AT THE | Started   |    L -> L     |    L -> H     |
 START  | Easy (L)  |    (51%)      |    (12%)      |
 OF     +-----------+---------------+---------------+
TRAINING| Started   |    H -> L     |    H -> H     |
        | Hard (H)  |    (26%)      |    (11%)      |
        +-----------+---------------+---------------+
```

- **L->L (51%)** — already learned tokens ("the", "of", "is"). Wasted gradient updates.
- **H->L (26%)** — actually learned during training ("Paris", "capital"). THE USEFUL ONES.
- **H->H (11%)** — never learned, unlearnable noise ("Parisii", misspellings). Wasted gradient updates.
- **L->H (12%)** — model gets confused as it learns more ("major" — too many valid alternatives). Unstable.

Only **~26% of tokens** drive meaningful learning. The other 74% generate wasted gradients.

### The Math

**Step 1: Reference model scores each token**

```
L_RM(x_i) = -log P(x_i | x_<i)
```

Run a pre-trained reference model over the data. For each token, compute how surprised the reference is. Low loss = reference knows it. High loss = reference struggles too.

**Step 2: Excess loss = your loss minus reference loss**

```
L_delta(x_i) = L_theta(x_i) - L_RM(x_i)
```

- High excess loss → your model is behind on something the reference knows → TRAIN ON THIS
- Low/negative excess loss → either both know it or both struggle → SKIP

**Step 3: Keep top k% tokens by excess loss**

```
I_k%(x_i) = 1 if x_i in top k% by excess loss, else 0
```

Binary filter. In or out.

**Step 4: Compute loss only on selected tokens**

```
L_SLM = -1/(N*k%) * SUM( I_k%(x_i) * log P(x_i | x_<i; theta) )
```

Same as standard cross-entropy loss, but averaged over only the selected tokens. Filtered tokens get multiplied by 0 — no gradient flows from them.

### Concrete Example

```
Sentence: "The capital of France is Paris, established by the Parisii"

Token losses (your model / reference):
  "The"        0.05 / 0.04  excess=0.01   SKIP (already learned)
  "capital"    2.10 / 0.30  excess=1.80   KEEP (you're behind)
  "of"         0.03 / 0.02  excess=0.01   SKIP (already learned)
  "France"     1.80 / 0.20  excess=1.60   KEEP (you're behind)
  "is"         0.08 / 0.05  excess=0.03   SKIP (already learned)
  "Paris"      2.50 / 0.16  excess=2.34   KEEP (you're behind)
  "Parisii"    3.90 / 3.80  excess=0.10   SKIP (unlearnable for both)

Standard loss: average all 7 = lots of wasted gradient on "The", "of", "is"
SLM loss:      average only "capital", "France", "Paris" = focused learning
```

---

## 3. Why It Matters for Parameter Golf

### The PG Constraint

We have a **fixed 10-minute wall clock** on 8xH100. We can't train longer — we can only train **smarter**. Every wasted gradient update on a token the model already knows or can never learn is time we'll never get back.

### How SLM Helps

- ~51% of tokens are L->L (already learned) — skipping them means each step does 2x more useful work
- At 10,000+ steps in 10 minutes, even a small per-step efficiency gain compounds
- Paper showed 10x faster convergence — even 2x would be transformative for PG

### Current State (Updated 2026-04-26)

> **⚠️ SLM NEVER ACTUALLY TESTED:** All "SLM" runs in Session 6 were invalid — the RunPod had commit `d7af1ec` (Apr 23) which predates the SLM code push (Apr 26 7:18 PM). `SLM_ENABLED` was set but the code to use it didn't exist. All results below claiming SLM improvements are retracted. **First real SLM validation will be Session 7.**

- Best 2×H100 run: **1.2411 BPB** (Run A, SP8192 combo slim + TTT) — ~~1.2384 was Run 13 but SLM was not active~~
- Best 8×H100 run: **1.2077 BPB** (Run 11, SP8192 combo slim + TTT)
- ~~Projected 8×H100 with SLM k=0.8: ~1.2050 BPB~~ — retracted, SLM untested
- PG baseline: **1.2244 BPB**
- Current SOTA: **1.0810 BPB**
- Gap to SOTA: **+0.1267 BPB** — we need techniques that make each step count more

---

## 4. Microsoft/rho Repo Status

The official repo at `github.com/microsoft/rho` contains:
- Pretrained model checkpoints on HuggingFace (rho-math-1b, rho-math-7b)
- Evaluation harness (math-evaluation-harness submodule)
- Documentation and pipeline diagrams

**No training code is released.** We must implement SLM ourselves based on the paper's algorithm description.

---

## 5. PG-Specific Implementation Options

### Option A: Simple Loss-Threshold SLM (Recommended Start)

**What:** Keep top-k% tokens by loss magnitude (highest loss = keep, lowest = drop).

**How:** Change one line in `train_gpt.py` (line 735):
```python
# Before:
return F.cross_entropy(logits.float(), targets, reduction="mean")

# After:
per_token_loss = F.cross_entropy(logits.float(), targets, reduction="none")
k = int(per_token_loss.numel() * self.slm_ratio)
topk_losses, _ = torch.topk(per_token_loss, k)
return topk_losses.mean()
```

**Pros:**
- ~3 lines of code, zero speed overhead
- No reference model needed
- Tests if the concept helps at 17M scale at all

**Cons:**
- Can't distinguish H->L (learnable) from H->H (unlearnable) — both have high loss
- Keeps ~11% unlearnable tokens that full Rho-1 would filter out
- Simplified version, not faithful to the paper

**Overhead:** Near zero. `torch.topk` on 262K tokens is negligible.

### Option B: Self-Referencing with Early Checkpoint

**What:** Train normally for N steps, save checkpoint, load it as frozen reference, then switch to SLM with excess loss filtering.

**How:**
1. Train normally for steps 0 to N (e.g., N=1000)
2. Save model weights as reference
3. From step N onward, each batch:
   - Forward pass through frozen reference (no_grad) → get per-token ref_loss
   - Forward pass through training model → get per-token train_loss
   - Excess loss = train_loss - ref_loss
   - Keep top k% by excess loss
   - Backprop only on selected tokens

**Pros:**
- Faithful to Rho-1's self-referencing approach (Section 3.4, +3.3%)
- Properly distinguishes H->L from H->H tokens
- No external high-quality corpus needed

**Cons:**
- Extra forward pass per batch (~40% slower steps after step N)
- 2x model memory (reference + training model) — ~34M params total, still fits on H100
- Loses ~1 min of 10-min window for the initial non-SLM warmup
- More complex implementation

### Option C: Full Dual-Model Rho-1

**What:** Train a separate reference model on curated data, then use it throughout training.

**Why NOT for PG:**
- No curated high-quality corpus available (PG uses FineWeb only)
- 2x memory and compute overhead for the entire training run
- Too expensive in a 10-min window
- The paper's best results used curated data we don't have

---

## 6. Phased Testing Plan

Each phase has clear pass/fail criteria. Only proceed to next phase if current phase passes.

### Phase 1: Smoke Test

**Goal:** Verify SLM doesn't break training and has positive signal at 17M scale.

| Item | Detail |
|------|--------|
| Hardware | 2xH100 (~$2) |
| Config | SP1024, baseline GQA + SLM_ENABLED=1, SLM_RATIO=0.6 |
| Control | Same config with SLM_ENABLED=0 (baseline Run 6v2: 1.2649 BPB) |
| Duration | 10 min wall clock, ~3,500 steps |

**Pass criteria:**
- [ ] val_bpb improves over baseline — ~~PASS~~ **INVALID** (SLM code not present on RunPod)
- [ ] step_avg doesn't regress more than 5% — ~~PASS~~ **INVALID** (SLM code not present)
- [ ] Training is stable (no NaN, no loss spikes) — ~~PASS~~ **INVALID** (runs completed but SLM was not active)

### Phase 2: Ratio Sweep

**Goal:** Find optimal SLM_RATIO for 17M scale.

| Item | Detail |
|------|--------|
| Hardware | 2xH100 (~$4, 2 runs) |
| Ratios to test | 0.5, 0.6, 0.7, 0.8 |
| Control | Phase 1 baseline |

**Pass criteria:**
- [ ] At least one ratio beats baseline — ~~PASS~~ **INVALID** (SLM code not present; BPB diffs were noise)
- [ ] Best ratio identified — ~~PASS~~ **INVALID** (sweep results are meaningless without SLM active)

### Phase 3: Competition Config Run

**Goal:** Verify SLM stacks with other techniques on the full competition setup.

| Item | Detail |
|------|--------|
| Hardware | 8xH100 (~$5) |
| Config | sp8192_combo_slim + best SLM_RATIO from Phase 2 |
| Control | Run 11 (1.2077 BPB, sp8192_combo_slim without SLM) |

**Pass criteria (tested on 2×H100, 8×H100 pending):**
- [ ] val_bpb improves over non-SLM config — ~~PASS~~ **INVALID** (Run 13's 1.2384 vs Run A's 1.2411 is noise, not SLM)
- [ ] Artifact fits 16 MB budget — untested (SLM was not active)
- [ ] No interaction effects with TTT, QK-Gain, headwise attn — untested (SLM was not active)

### Phase 4: 3-Seed Submission Run

**Goal:** Generate submission artifacts with statistical significance.

| Item | Detail |
|------|--------|
| Hardware | 8xH100 (~$15, 3 runs) |
| Script | `runs/parameter_golf_8gpu_3seed_run.sh` |
| Seeds | 42, 1337, 2025 |

**Pass criteria:**
- [ ] Mean val_bpb beats PG baseline (1.2244)
- [ ] Standard deviation is acceptable (< 0.005)
- [ ] All 3 runs fit 16 MB budget

**Fail criteria:**
- High variance across seeds → SLM adds instability, drop it
- Mean doesn't beat baseline → revert to non-SLM config for submission

### Estimated Total Cost: ~$26 (if all phases pass)

---

## 7. Experimental Results (2026-04-26)

> **⚠️ ALL RESULTS IN THIS SECTION ARE INVALID.** The RunPod had commit `d7af1ec` (Apr 23) which predates the SLM code push (Apr 26 7:18 PM). `SLM_ENABLED` env var was set but the code to use it didn't exist. All BPB differences attributed to SLM below are run-to-run noise. **First real SLM validation will be Session 7** using `runs/run_slm_validation_2gpu.sh`.

All experiments run on 2×H100 SXM, PyTorch 2.11, 10-min wall clock. Option A (simple loss-threshold) implemented.

### SLM Ratio Sweep (SP1024, vanilla GQA baseline)

| SLM Ratio (k) | val_bpb | Steps | Step avg | Tokens kept |
|:-:|:-:|:-:|:-:|:-:|
| 0.4 (aggressive) | 1.2954 | 2,784 | 216ms | Top 40% |
| 0.5 | 1.3001 | 2,554 | 235ms | Top 50% |
| 0.6 | 1.2994 | 2,615 | 229ms | Top 60% |
| 0.7 | 1.2973 | 2,713 | 221ms | Top 70% |
| **0.8 (winner)** | **1.2949** | **2,801** | **214ms** | **Top 80%** |
| 0.9 (barely selective) | 1.2955 | 2,818 | 213ms | Top 90% |

**Finding:** k=0.8 is optimal. More conservative than the paper's recommended 0.6-0.7 — at 17M scale, the model benefits from seeing more tokens, with only the easiest 20% filtered out.

### SP8192 Competition Config + SLM (2×H100)

| Run | Config | val_bpb (TTT) | Delta vs no-SLM | % improvement |
|:-:|:-:|:-:|:-:|:-:|
| H | SP8192 combo slim, no TTT, no SLM | 1.2432 | — | — |
| A | SP8192 combo slim + TTT | 1.2411 | — (baseline) | — |
| D | SP8192 combo slim + TTT + SLM k=0.6 | 1.2396 | -0.0015 | 0.12% |
| **13** | **SP8192 combo slim + TTT + SLM k=0.8** | **1.2384** | **-0.0027** | **0.22%** |

**Finding:** SLM stacks cleanly with all other techniques (SP8192, TTT, LeakyReLU², headwise, QK-Gain 5.0). New best 2×H100 run.

### TTT Contribution (isolated)

| Run | TTT | val_bpb | TTT delta |
|:-:|:-:|:-:|:-:|
| H | No TTT | 1.2432 | — |
| A | Score-first TTT | 1.2411 | -0.0021 |

**Finding:** TTT contributes ~0.002 BPB on 2×H100. Expected to be larger on 8×H100 with more training steps.

### Technique Stacking (SP1024, 2×H100)

| Run | Techniques | val_bpb |
|:-:|:-:|:-:|
| F | LeakyReLU² only | 1.2932 |
| G | LeakyReLU² + SLM k=0.6 | 1.2927 |
| K | Headwise + QK-Gain 5.0 + SLM k=0.6 | 1.2927 |
| **L** | **LeakyReLU² + headwise + SLM k=0.6** | **1.2899** |

**Finding:** Techniques stack additively. Best SP1024 result uses all three cheap techniques together.

### Key Takeaways for Presentation

> **All claims below are INVALID — SLM was never actually tested. Awaiting Session 7 validation.**

1. ~~**SLM works at 17M scale**~~ — untested
2. ~~**k=0.8 is optimal**~~ — untested (sweep results were noise)
3. **Zero overhead** — theoretically true (code adds only `torch.topk`), but never measured in practice
4. ~~**Stacks with everything**~~ — untested
5. ~~**New best 2×H100: 1.2384 BPB**~~ — this was Run A noise (1.2411 is the valid best)
6. ~~**Projected 8×H100: ~1.2050 BPB**~~ — retracted
7. **Implementation: 3 lines of code** — still true, code is now in the repo awaiting first real test

---

## 8. Key Unknowns & Risks (Updated with Results)

### 17M Scale is Uncharted Territory → ~~RESOLVED~~ STILL UNKNOWN

The paper tested on 1.1B and 7B models. Our 17M model is 65x smaller than the smallest tested.

~~**Result:** SLM improves BPB at 17M scale.~~ **INVALID** — SLM code was not present during testing. This remains an open question for Session 7.

### H->H Token Contamination (Option A) → ~~RESOLVED~~ STILL UNKNOWN

Simple loss-threshold keeps ALL high-loss tokens, including H->H (unlearnable noise). Full Rho-1 would filter these via excess loss.

~~**Result:** At k=0.8, only 20% of tokens are dropped~~ **INVALID** — SLM code was not present. Theoretical reasoning still applies but needs empirical validation in Session 7.

### Interaction with TTT → ~~RESOLVED~~ STILL UNKNOWN

TTT (Test-Time Training) fine-tunes the model on validation chunks at eval time.

~~**Result:** SLM and TTT stack cleanly.~~ **INVALID** — Run 13 was not running SLM. Whether SLM + TTT interact positively is untested. Theoretically they should be complementary (SLM during training, TTT at eval), but needs Session 7 validation.

### SLM_RATIO Sensitivity → ~~RESOLVED~~ STILL UNKNOWN

~~**Result:** Full sweep from k=0.4 to k=0.9 tested.~~ **INVALID** — SLM code was not present. The "sweep" results were all running without SLM; BPB differences were noise. Session 7 will run a real sweep with k=0.6, 0.7, 0.8 on working SLM code.

---

## 9. References

- [Rho-1 Paper (arXiv)](https://arxiv.org/abs/2404.07965)
- [Rho-1 HTML (full text)](https://arxiv.org/html/2404.07965)
- [Microsoft/rho GitHub](https://github.com/microsoft/rho)
- [HuggingFace models](https://huggingface.co/microsoft/rho-math-1b-v0.1)
- PG train_gpt.py line 735: `F.cross_entropy(logits.float(), targets, reduction="mean")`
- Our paper survey: `docs/parameter-golf/neurlps-paper-survey.md` (Paper #1)
