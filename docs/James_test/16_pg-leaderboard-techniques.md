# Parameter Golf Leaderboard — Technique Breakdown

**Author:** James Vo
**Date:** 2026-04-22
**Source:** READMEs from `parameter-golf/records/track_10min_16mb/`, leaderboard in `parameter-golf/README.md`

## TL;DR

The PG leaderboard went from 1.2244 (baseline) to 1.0810 (SOTA) in 23 days via 28 submissions. The final SOTA stacks **7 major techniques** on top of each other. Each technique gave a measurable BPB drop on its own, but they compound non-linearly when combined. This note explains each technique in plain English so teammates can understand what the top entries actually did.

---

## Progression Table — Key Milestones

The full leaderboard has 28 entries. This table shows the major milestones where a new technique class first appeared:

| Date | Entry | BPB | Delta from Baseline | Key New Technique |
|------|-------|-----|---------------------|-------------------|
| Mar 17 | Naive Baseline | 1.2244 | -- | 9L/512d, 1024 vocab, int8+zlib |
| Mar 19 | Sliding Window Eval | 1.1925 | -0.0319 | Stride-64 sliding window eval |
| Mar 19 | Mixed Quant + Sliding Window | 1.1630 | -0.0614 | Int6 blocks + int8 embeddings |
| Mar 20 | Efficient Partial XSA | 1.1307 | -0.0937 | Extra-State Attention on deep layers |
| Mar 22 | EMA + GPTQ-lite + Warmdown | 1.1228 | -0.1016 | EMA averaging + GPTQ + warmdown |
| Mar 23 | LeakyReLU² + Legal TTT | 1.1194 | -0.1050 | Score-first test-time training |
| Mar 25 | AR Self-Gen GPTQ + XSA | **1.1147** | -0.1097 | Full-Hessian GPTQ (1st merged SOTA) |
| Mar 31 | Parallel Residuals + Mini DR | 1.1063 | -0.1181 | Parallel residuals + depth recurrence |
| Apr 01 | 4096-Vocab + High WD | 1.0979 | -0.1265 | Vocab scaling (1024 -> 4096) |
| Apr 03 | MuonEq-R + DR + All-Int6 | 1.0912 | -0.1332 | Row-normalized Muon optimizer |
| Apr 05 | SP8192 + GPTQ Embeds + SDClip | 1.0856 | -0.1388 | Vocab scaling (4096 -> 8192) |
| Apr 06 | SP8192 + QK5 + Legal TTT | 1.0828 | -0.1416 | QK-Gain tuning + TTT on SP8192 |
| Apr 08 | SP8192 + ParResid + TTT | 1.0822 | -0.1422 | Combined parallel resid + TTT |
| Apr 09 | SP8192 + 3L Recur + ParResid | **1.0810** | **-0.1434** | 3-layer recurrence + QK 5.25 (SOTA) |

Total improvement: **0.1434 BPB** (11.7% relative reduction from baseline).

---

## ASCII Progression

```
BPB
1.225 | X Baseline (Mar 17)
1.200 |  \
1.175 |   \
1.150 |    \______ Sliding window + quant + XSA era (Mar 19-25)
1.125 |           \
1.100 |            \__ Parallel Resid + Depth Recurrence (Mar 31)
1.075 |               \__ Vocab scaling era: SP4096, SP8192 (Apr 1-5)
1.050 |                  \_ TTT + QK-Gain + final stacking (Apr 6-9)
1.025 |
1.000 |
      +----+----+----+----+----+----+-----> Date
         Mar17 Mar20 Mar25 Mar31 Apr01 Apr05 Apr09

Three eras:
  Era 1 (Mar 17-25): Eval & quantization tricks       -0.1097 BPB
  Era 2 (Mar 31-Apr 3): Architecture innovations       -0.0235 BPP
  Era 3 (Apr 5-9): Vocab scaling + technique stacking  -0.0102 BPB
```

---

## Technique Deep Dives

### 1. SP8192 — Vocabulary Scaling (1024 -> 8192)

**What it does:** Replaces the tiny 1024-token SentencePiece BPE vocabulary with an 8192-token one. More tokens means the model can represent common word fragments as single tokens instead of spelling them out character by character.

**Why it helps:** With 1024 tokens, the model burns capacity encoding frequent subwords like "tion", "ing", "the" as multi-token sequences. With 8192 tokens, these become single tokens. The model spends fewer steps per word and can focus capacity on learning actual language patterns. It also means fewer total tokens in each sequence, so the model sees more "content" per training step.

**Progression:**
- SP1024 (baseline): 1.2244 BPB
- SP4096 (Apr 1, @clarkkev): 1.0979 BPB — first vocab jump
- SP8192 (Apr 5, @clarkkev PR #1394): 1.0856 BPB — the final vocab

**Measured gain:** ~0.012 BPB going from SP4096 to SP8192 (within the same technique stack). Vocab scaling from 1024 to 4096 was worth ~0.008 BPB on its own when first introduced.

**Trade-off:** Larger vocab means larger embedding tables, which eat into the 16 MB budget. GPTQ int8 embedding quantization (see technique 7) makes this feasible.

> **Nanochat:** Uses 32K vocab (different scale). No direct comparison, but confirms bigger vocab = better.

---

### 2. Depth Recurrence — Reusing Layers Without Extra Parameters

**What it does:** Instead of adding more unique layers (which would cost more parameters), the model runs the same physical layers multiple times. For example, physical layers 3, 4, 5 are run, then looped back to run 3, 4, 5 again. This gives 17 "virtual" layers from only 11 physical layers — zero extra parameters.

**Why it helps:** Deeper networks learn more abstract representations. Each pass through the same layers refines the representation further, similar to how humans re-read a complex sentence to understand it better. The key insight is that recurrence through a few middle layers is more effective than trying to recur the whole stack.

**How it works in practice:**
- Encoder path: layers [0, 1, 2, 3, 4, 5, 3, 4]
- Decoder path: layers [5, 3, 4, 5, 6, 7, 8, 9, 10]
- Layers 3-5 are the "recurrence block" — they run 3 times total
- Recurrence activates mid-training (at ~35-50% of steps) so the model first learns basic representations without the slower recurrence overhead
- The repeated MLPs can optionally be "untied" (given separate weights) for a small improvement

**Progression:**
- No recurrence (baseline): 1.2244 BPB
- Mini depth recurrence, layers 4-5, 2 loops (Mar 31, @msisovic): contributed to 1.1063 entry
- 3-layer recurrence, layers 3-4-5 (Apr 9, @dexhunter PR #1331, #1437): part of 1.0810 SOTA

**Measured gain:** ~0.002-0.004 BPB. Small per-entry but compounds well with parallel residuals.

> **Nanochat:** Not tested. This is a PG-specific innovation.

---

### 3. Parallel Residuals — Separate Data Paths for Attention and MLP

**What it does:** Normally in a transformer, each layer runs attention first, then feeds the result into the MLP. With parallel residuals (GPT-J style), from layer 7 onward, attention and MLP operate on **separate copies** of the input simultaneously. A learned scalar blends their outputs back together.

**Why it helps:** When attention and MLP read from different "residual lanes," attention can specialize on context mixing (which tokens relate to which) while MLP specializes on token transformations (what does this token mean). This decoupling lets each sub-layer become more efficient at its specific job.

**Interesting detail:** The learned routing becomes asymmetric — MLP barely writes back into attention's residual stream, especially in deeper layers. This suggests that once attention has done its context-gathering job, the MLP doesn't need to disturb it.

**First appeared:** Mar 31 (@msisovic PR #1204, concept) / (@Robby955 PR #1412, on SP8192)

**Measured gain:** ~0.002-0.004 BPB individually. Worth ~0.0022 BPB on top of a depth-recurrence stack.

> **Nanochat:** Not tested. PG innovation.

---

### 4. QK-Gain — Learnable Attention Scaling

**What it does:** Standard attention divides query-key dot products by sqrt(d_head) to prevent large values. QK-Gain replaces this fixed scaling with a **learnable per-head parameter** that the model tunes during training. The optimal value turned out to be 5.25 (vs. the standard ~4.0 from sqrt(64)=8, or the manually-set 4.0 starting point).

**Why it helps:** Different attention heads learn different patterns — some attend broadly, others narrowly. A fixed scaling factor forces all heads to use the same "temperature" for their attention distributions. Learnable scaling lets each head choose its own sharpness, making attention more expressive without adding any meaningful parameter count.

**Progression:**
- QK-Gain 4.0 (@clarkkev PR #1394): part of 1.0856
- QK-Gain 5.0 (@dexhunter): improved to 1.0828 (Apr 6)
- QK-Gain 5.25 (@bigbag): part of 1.0810 SOTA (Apr 9)

**Measured gain:** ~0.003 BPB going from 4.0 to 5.0 (on the SP8192 stack). Monotonic improvement from 4.0 to 5.25.

> **Nanochat:** Tested simpler per-head attention gates (Jan 17) — no improvement. QK-Gain is different (scaling, not gating).

---

### 5. MuonEq-R — Row-Normalized Muon Optimizer

**What it does:** Muon is an optimizer that uses Newton-Schulz iterations to orthogonalize gradient matrices (making weight updates more efficient). MuonEq-R adds **row normalization** — each row of the gradient matrix is normalized before the orthogonalization step. This ensures all rows contribute equally to the update, preventing some neurons from dominating the learning.

**Why it helps:** In standard Muon, rows with larger gradient norms get disproportionately large updates, which can cause training instability or leave some neurons under-trained. Row normalization acts like a per-neuron learning rate equalizer, leading to more stable and uniform convergence.

**Reference:** arXiv:2603.28254

**First appeared:** Apr 3 (@dexhunter PR #1260): 1.0912 BPB entry

**Measured gain:** Part of the ~0.007 BPB improvement from 1.0979 to 1.0912 (combined with depth recurrence and all-int6 GPTQ).

> **Nanochat:** Uses Muon (Polar Express variant). Row-normalization not tested. Ran ~320 optimizer experiments and found "hyperparams are scale-dependent" — d12 tuning hurts d20.

---

### 6. Score-First TTT — Legal Test-Time Training

**What it does:** After training is done, the model adapts itself on the validation data — but in a strictly legal way. The validation tokens are split into 32K-token chunks. For each chunk: (1) score all tokens under `torch.no_grad()` (no weight updates), (2) only then train on the tokens you already scored using SGD. This means the model never "cheats" by training on tokens before grading them.

**Why it helps:** Language has local patterns — if the first paragraph of a document is about physics, the rest probably is too. TTT lets the model adjust to the local distribution of the text it's currently evaluating, improving predictions on later tokens. It's like giving the model a few minutes to "study" each test question's topic before answering.

**Key parameters:** SGD with lr=0.005, momentum=0.9, 3 epochs per chunk, cosine LR decay across chunks.

**Compliance rules (Issue #1017):**
1. Causality — each position scored from prefix tokens only
2. Normalized distribution — standard softmax, no logit biasing
3. Score before update — every chunk scored BEFORE any weight update
4. Single pass — each token scored exactly once

**First appeared:** Mar 23 (@abaybektursun PR #549): 1.1194 BPB (first legal TTT entry)

**Measured gain:** ~0.002-0.003 BPB on top of the SP8192 stack. For example, on the Apr 6 entry: sliding BPB 1.0849 -> TTT BPB 1.0828 = -0.0021 BPB gain.

> **Nanochat:** Not tested. TTT is eval-time only; nanochat focuses on training speed.

---

### 7. GPTQ Embeddings + SDClip — Smart Quantization

**What it does:** After training, model weights are compressed using GPTQ (a Hessian-aware quantization method). The innovation here is twofold:

1. **GPTQ for embeddings** — token embedding tables are quantized to int8 (not just the attention/MLP matrices which use int6). This saves crucial bytes when using large vocabs (8192 tokens).
2. **SDClip** — instead of clipping outlier weights at a fixed threshold, clipping is set to `k * std(row)` where k is tuned per-layer (k=12.85 for matrices, k=20.0 for embeddings). This preserves important outliers that carry disproportionate information.

**Why it helps:** The 16 MB budget is tight. Without int8 embedding quantization, an 8192-token vocab wouldn't fit. SDClip reduces quantization error compared to naive clipping, preserving model quality after compression. The combination costs only ~0.001 BPB in quality loss.

**Compression pipeline:** Full-Hessian GPTQ -> int6 matrices + int8 embeddings -> byte-shuffle -> Brotli-11 compression -> LZMA self-extracting wrapper (~16.6 KB code overhead).

**First appeared as a system:** Apr 5 (@clarkkev PR #1394): 1.0856 BPB

> **Nanochat:** No 16 MB constraint. Quantization not applicable to nanochat's goals.

---

### 8. Hyperparameter Tuning

**What it does:** Careful tuning of training hyperparameters beyond their defaults:

| Parameter | Default/Early | Final Tuned | What It Controls |
|-----------|--------------|-------------|------------------|
| Weight decay (WD) | 0.04 | 0.095 | How aggressively unused weights are pushed toward zero |
| Max learning rate (MLR) | ~0.01 | 0.022 | Peak learning rate during training |
| EMA decay | ~0.99 | 0.9965 | How slowly the exponential moving average model updates |
| Warmdown fraction | ~0.50 | 0.72 | What fraction of training uses linear LR decay to zero |

**Why it helps:** Higher weight decay (0.095 vs 0.04) acts as a stronger regularizer, preventing overfitting during the short 10-minute training window. The long warmdown (72% of training) means the model spends most of its time in a "cooling" phase where learning rate gradually drops, allowing fine-grained convergence. Higher EMA decay (0.9965) means the final averaged model incorporates more of the training history.

**Contributor:** @X-Abhishek-X (PR #1445, #1471)

> **Nanochat:** Ran ~320 optimizer experiments (Jan 19-22). Key finding: "what works at d12 doesn't transfer to d20." PG is ~d8-d10 scale, so small-scale nanochat results are most relevant to us.

---

## Key Insight: Non-Linear Compounding

The techniques don't add up linearly. Here's what happens when you stack them:

```
Technique                        Individual Gain    Stacked Gain
--------------------------------------------------------------
SP8192 (vocab)                     ~0.012 BPB        0.012
+ Depth recurrence                 ~0.003 BPB        0.016
+ Parallel residuals               ~0.003 BPB        0.020
+ MuonEq-R                         ~0.004 BPB        0.025
+ QK-Gain tuning                   ~0.003 BPB        0.029
+ GPTQ Embeddings + SDClip         ~0.004 BPB        0.034
+ Score-first TTT                  ~0.002 BPB        0.034 (eval)
+ Hyperparameter tuning            ~0.003 BPB        0.034+

Sum of individual gains:           ~0.034 BPB
Actual total improvement:           0.1434 BPB  (4.2x more than sum!)
```

The 4.2x multiplier comes from the fact that each technique amplifies the others. Better architecture (depth recurrence, parallel residuals) gives the optimizer (MuonEq-R) more structure to exploit. Better quantization (GPTQ+SDClip) preserves more of the quality gained from architectural improvements. Larger vocab (SP8192) benefits more from deeper models than smaller vocab does. TTT compounds with all of the above because a better-trained model adapts more effectively at test time.

**This is why ablation studies are essential** — you can't predict the combined gain from individual gains.

---

## Implications for Our Experiments

### Priority order for ablation

Based on magnitude of individual gains and ease of implementation:

1. **Vocab scaling** (SP4096 or SP8192) — largest single gain, straightforward to try
2. **Depth recurrence** — zero extra params, just loop middle layers
3. **Parallel residuals** — moderate code change, well-documented
4. **Hyperparameter sweep** — WD, MLR, EMA, warmdown — no code changes needed
5. **QK-Gain** — single env var change (`QK_GAIN_INIT=5.0`)
6. **TTT** — already implemented in PG codebase, just set `TTT_ENABLED=1`
7. **MuonEq-R** — optimizer change, moderate complexity
8. **GPTQ+SDClip** — quantization pipeline, highest complexity

### What's already in the codebase

All of these techniques are already implemented in the SOTA `train_gpt.py` and controlled via environment variables:

```bash
# Vocab
--variant sp8192  (data download)

# Depth recurrence
RECUR_LAYERS=3,4,5  RECUR_START_STEP=2016

# Parallel residuals
PARALLEL_START_LAYER=7  PARALLEL_RESIDUAL=1

# QK-Gain
QK_GAIN_INIT=5.25

# TTT
TTT_ENABLED=1  TTT_LR=0.005  TTT_EPOCHS=3

# Hyperparameters
WD=0.095  MLR=0.022  EMA=0.9965
```

### Ablation strategy

To measure each technique's contribution, run the full SOTA stack and then remove techniques one at a time:

```
Ablation 1: Full stack (expect ~1.081)
Ablation 2: Remove TTT (expect ~1.083)
Ablation 3: Remove depth recurrence (expect ~1.084)
Ablation 4: Remove parallel residuals (expect ~1.085)
Ablation 5: SP4096 instead of SP8192 (expect ~1.090)
Ablation 6: Default hyperparams (expect ~1.093)
Ablation 7: Remove QK-Gain (expect ~1.086)
```

Each ablation costs ~$2.50 on RunPod (10 min on 8xH100). Full ablation suite: ~$17.50.

### Skip list — nanochat already disproved

These techniques were tested in nanochat's `dev/LOG.md` and failed. Don't waste GPU credits re-testing:

| Technique | Nanochat Verdict | Skip Reason |
|-----------|-----------------|-------------|
| SwiGLU | Worse than ReLU² at d12 and d24 | Same architecture, same ancestry |
| MoE (Mixture of Experts) | Net negative — grouped_mm overhead killed throughput | Even worse at 17M params |
| Hyperball/MuonH | Worse across all LR sweeps | Muon already works |
| Partial RoPE (1/4 dims) | Slightly worse | Full RoPE is better |
| LN Scale (1/√layer) | Did not help | Not worth complexity |
| Orthogonal init | Did not help | PG's zero-init + default works |
| Multi-Token Prediction | Worse wall clock, +13 GB memory | Overhead dominates at 17M |
| Varlen attention | 0.0002 BPB (noise) | Not worth torch.compile hassle |

**Caveat:** Nanochat optimizes for wall clock time at ~d24 scale. PG optimizes for BPB at ~d8-d10 scale. A technique that failed for speed might still help BPB — but the clear negatives above (SwiGLU, MoE, MuonH) failed on quality too, not just speed.

### Worth testing — nanochat supports

These had positive or mixed results at PG-relevant scale:

| Technique | Nanochat Verdict | Why Test for PG |
|-----------|-----------------|-----------------|
| LeakyReLU² | Better per-step quality, slower wall clock | PG cares about BPB, not speed |
| BigramHash | Adopted at d12 (our scale), reverted at d25 | PG rank 1 uses it. Works at our model size. |
| SoftCap=20 | Optimal in nanochat (vs default 30) | Trivial env var: `LOGIT_SOFTCAP=20` |
| XSA | Better step quality, not wall clock | Quality > speed for PG |
| Gated Attention (NeurIPS 2025) | Simpler per-head gates failed, but Q-projection sigmoid is different and untested | Already coded in our `train_gpt.py` |

---

## Cross-Reference

- `13_pg-vs-nanochat-architecture.md` — Architectural comparison (PG vs nanochat share most techniques)
- `14_modded-nanogpt-lineage.md` — PG descends from modded-nanogpt, already has the key techniques
- `17_pg-leaderboard-annotated.md` — All 30 leaderboard entries with component-tagged techniques
- `docs/parameter-golf/findings.md` — Our actual run results (Run 1: val_bpb 1.3045 on 2 GPUs)
- `nanochat/dev/LOG.md` — Nanochat experiment log (Jan–Mar 2026, ~30 experiments)

## Sources

- `parameter-golf/README.md` — Full leaderboard table
- `parameter-golf/records/track_10min_16mb/2026-04-09_*/README.md` — SOTA entry (1.0810)
- `parameter-golf/records/track_10min_16mb/2026-04-08_*/README.md` — ParResid + TTT (1.0822)
- `parameter-golf/records/track_10min_16mb/2026-04-06_SP8192_QK5_*/README.md` — QK5 + TTT (1.0828)
- `parameter-golf/records/track_10min_16mb/2026-04-04_*/README.md` — SP4096 + DR + MuonEq-R (1.0897)
- `parameter-golf/records/track_10min_16mb/2026-03-31_*/README.md` — Parallel Resid + Mini DR (1.1063)
- `parameter-golf/records/track_10min_16mb/2026-03-17_*/README.md` — Naive Baseline (1.2244)
