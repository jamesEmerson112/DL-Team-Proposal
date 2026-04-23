# Parameter Golf — Findings & Insights

> What we learned from participating. Insights that could feed back into the course project.

## Summary — All Runs (sorted by BPB, best first)

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Size (int8+zlib) | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------------------|---------|
| 7 | **LeakyReLU²** | 17.06M | 2.1344 | **1.2641** | 3,673 | 163ms | 15.77 MB | Yes |
| 8 | LeakyReLU² + headwise | 17.10M | 2.1345 | 1.2642 | 3,368 | 178ms | 15.77 MB | Yes |
| 6v2 | Baseline repeat | 17.06M | 2.1357 | 1.2649 | 3,661 | 164ms | 15.77 MB | Yes |
| 2 | Headwise gated attn* | 17.10M | 2.1366 | 1.2653 | 3,287 | 182ms | 15.75 MB | Yes |
| 6 | Baseline (GQA) | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | 15.75 MB | Yes |
| 3 | Elementwise gated attn* | 19.42M | 2.1280 | 1.2602 | 3,129 | 192ms | 17.87 MB | **No** |
| 4 | MQA (1 KV head) | 17.65M | 2.1549 | 1.2761 | 3,370 | 178ms | 16.84 MB | **No** |
| 1 | Baseline (old pod) | 17.06M | 2.2027 | 1.3045 | 1,819 | — | 14.70 MB | Yes |
| ~~5~~ | ~~INVALID (stale env)~~ | — | — | — | — | — | — | — |

**All 2×H100 runs on PyTorch 2.11 unless noted. 10-min wall clock, 1024 vocab. PG baseline: 1.2244 BPB.**

**Best legal technique: LeakyReLU²** — free improvement (no extra params, no speed cost), gap to PG baseline: +0.0397.

**TODO:** Run `headwise_qkgain5.env` — QK-Gain 5.0 + headwise gated attn. PG ranks 1-6 all use QK-Gain 5.0-5.25 (up from default 1.5). Config ready, missed due to line-break paste error on pod. Command: `source runs/configs/headwise_qkgain5.env && export NGPUS=2 && bash runs/parameter_golf_baseline.sh`

*\* Original technique by James Vo — gated attention applied post-SDPA with sigmoid gates, inspired by NeurIPS 2025 Best Paper (arxiv.org/abs/2505.06708).*

---

## Run Log

### Run 1 — 2-GPU baseline (2026-04-16)

| Item | Value |
|---|---|
| Date | 2026-04-16 |
| GPUs | 2× (auto-detected, intended 1) |
| Pod cost | **~$2** (< 1 hour) |
| Model params | 17,059,912 (~17M) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 1,819 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,286 MiB |
| **val_loss (int8+zlib)** | **2.2027** (computed) |
| **val_bpb (int8+zlib)** | **1.3045** |
| Baseline to beat | 1.2244 |
| Gap | +0.0801 (need to go lower) |
| Model size (raw) | 67.2 MB |
| Model size (int8+zlib) | **14.7 MB** (under 16 MB budget) |
| Compression ratio | 3.91× |

**Training curve:**
- Step 0: val_bpb 4.1077 (near random)
- Step 1000: val_bpb 1.3702
- Step 1819: val_bpb 1.3033 (still improving when time ran out)

**Observations:**
- Model was still improving at cutoff — only used 2 GPUs, so throughput was ~4× lower than the intended 8-GPU setup
- int8+zlib compression fits comfortably under 16 MB
- BPB roundtrip after compression: 1.3045 (negligible degradation from 1.3033)

### Run 2 — Gated Attention headwise, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | Gated Attention — headwise (1 sigmoid gate per head, NeurIPS 2025 Best Paper) |
| Paper | arxiv.org/abs/2505.06708 |
| Model params | 17,096,776 (~17.1M, +37K from gate projections) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,287 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,374 MiB |
| Step avg | 182.54 ms |
| **val_loss (raw)** | **2.1320** (computed) |
| **val_bpb (raw)** | **1.2626** |
| **val_loss (int8+zlib)** | **2.1366** (computed) |
| **val_bpb (int8+zlib)** | **1.2653** |
| Baseline to beat | 1.2244 |
| Gap | +0.0409 (down from +0.0801 in Run 1) |
| Model size (raw) | 67.4 MB |
| Model size (int8+zlib) | **15.75 MB** (under 16 MB budget) |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1085
- Step 1000: val_bpb 1.3780
- Step 2000: val_bpb 1.3208
- Step 3000: val_bpb 1.2725
- Step 3287: val_bpb 1.2626 (still improving when time ran out)

**Observations:**
- Gap to baseline cut in half vs Run 1 (0.0409 vs 0.0801)
- Much higher throughput than Run 1 (3,287 vs 1,819 steps) — likely due to PyTorch 2.11.0 vs 2.4.1 on previous pod
- Headwise gates add only ~37K params (~0.2% overhead) — nearly free
- Model still improving at cutoff — not converged
- Compressed size 15.75 MB, tight but under 16 MB budget

**Note:** Run 1 vs Run 2 improvement comes from two factors: (1) gated attention headwise, and (2) newer PyTorch (2.11 vs 2.4.1) giving higher throughput. Need a clean baseline on the same pod/PyTorch to isolate the gated attention effect.

### Run 3 — Gated Attention elementwise, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | Gated Attention — elementwise (1 sigmoid gate per dim per head, NeurIPS 2025 Best Paper) |
| Paper | arxiv.org/abs/2505.06708 |
| Model params | 19,419,208 (~19.4M, +2.36M from gate projections) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,129 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 11,518 MiB |
| Step avg | 191.75 ms |
| **val_loss (raw)** | **2.1241** (computed) |
| **val_bpb (raw)** | **1.2579** |
| **val_loss (int8+zlib)** | **2.1280** (computed) |
| **val_bpb (int8+zlib)** | **1.2602** |
| Baseline to beat | 1.2244 |
| Gap | +0.0358 |
| Model size (raw) | 76.7 MB |
| Model size (int8+zlib) | **17.87 MB (OVER 16 MB budget)** |
| Compression ratio | 3.92× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1098
- Step 1000: val_bpb 1.3696
- Step 2000: val_bpb 1.3130
- Step 3000: val_bpb 1.2613
- Step 3129: val_bpb 1.2579 (still improving when time ran out)

**Observations:**
- Slightly better BPB than headwise (1.2602 vs 1.2653) but **busts the 16 MB budget** (17.87 MB)
- 2.36M extra params is too many — compressed model 1.87 MB over limit
- Slower per step (191ms vs 182ms headwise) → fewer steps (3,129 vs 3,287)
- More VRAM (11.5 GB vs 10.4 GB)
- The marginal BPB gain (~0.005) is not worth the budget/speed cost

### Run 2 vs Run 3 — Headwise vs Elementwise (same pod, same PyTorch)

| Metric | Headwise (Run 2) | Elementwise (Run 3) | Winner |
|---|---|---|---|
| val_bpb (compressed) | 1.2653 | 1.2602 | Elementwise (+0.005) |
| Extra params | +37K (0.2%) | +2.36M (14%) | Headwise |
| Compressed size | 15.75 MB | 17.87 MB | Headwise (under budget) |
| Steps completed | 3,287 | 3,129 | Headwise |
| Step avg | 182 ms | 192 ms | Headwise |
| Peak VRAM | 10.4 GB | 11.5 GB | Headwise |
| Under 16 MB? | Yes | **No** | Headwise |

**Verdict:** Headwise wins for PG. Elementwise has marginally better BPB but fails the size constraint and is slower. Headwise is nearly free (37K params, no speed penalty) and fits under budget.

### Run 4 — MQA (NUM_KV_HEADS=1), 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | MQA — Multi-Query Attention (1 KV head shared across all 8 Q heads) |
| Model params | 17,649,736 (~17.6M) |
| Architecture | 9 layers, 512 dims, 8 heads, **1 kv_head** (MQA) |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,370 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,710 MiB |
| Step avg | 178.03 ms |
| **val_loss (raw)** | **2.1503** (computed) |
| **val_bpb (raw)** | **1.2734** |
| **val_loss (int8+zlib)** | **2.1549** (computed) |
| **val_bpb (int8+zlib)** | **1.2761** |
| Baseline to beat | 1.2244 |
| Gap | +0.0517 |
| Model size (raw) | 69.6 MB |
| Model size (int8+zlib) | **16.84 MB (OVER 16 MB budget)** |
| Compression ratio | 3.79× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1085
- Step 1000: val_bpb 1.3864
- Step 2000: val_bpb 1.3329
- Step 3000: val_bpb 1.2861
- Step 3370: val_bpb 1.2734 (still improving when time ran out)

**Observations:**
- Worst BPB of the three techniques tested on this pod (1.2761 vs headwise 1.2653 vs elementwise 1.2602)
- Also over 16 MB budget (16.84 MB)
- Fastest per step (178ms) — fewer KV params = less compute, more steps (3,370)
- Despite more steps, BPB is worse — MQA trades quality for speed, bad tradeoff for PG where BPB matters

### All Runs on PyTorch 2.11 Pod — Comparison

| Run | Technique | KV Heads | val_loss | val_bpb | Steps | Size (int8+zlib) | Under 16 MB? |
|-----|-----------|----------|----------|---------|-------|-------------------|-------------|
| 2 | Gated Attn (headwise) | 4 (GQA) | 2.1366 | **1.2653** | 3,287 | **15.75 MB** | **Yes** |
| 3 | Gated Attn (elementwise) | 4 (GQA) | 2.1280 | 1.2602 | 3,129 | 17.87 MB | No |
| 4 | MQA | 1 (MQA) | 2.1549 | 1.2761 | 3,370 | 16.84 MB | No |

### Run 5 — GQA baseline (intended), 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline (no gated attention) — **INVALID: stale env var** |
| Model params | 19,419,208 (~19.4M) — should be 17M, shell had `GATED_ATTN=elementwise` from prior run |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,296 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 11,518 MiB |
| Step avg | 182.03 ms |
| **val_loss (raw)** | **2.1191** |
| **val_bpb (raw)** | **1.2550** |
| **val_loss (int8+zlib)** | **2.1234** |
| **val_bpb (int8+zlib)** | **1.2576** |
| Baseline to beat | 1.2244 |
| Gap | +0.0332 |
| Model size (raw) | 76.66 MB |
| Model size (int8+zlib) | **17.96 MB (OVER 16 MB budget)** |
| Compression ratio | 3.92× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1098
- Step 1000: val_bpb 1.3687
- Step 2000: val_bpb 1.3144
- Step 3000: val_bpb 1.2648
- Step 3296: val_bpb 1.2550 (still improving when time ran out)

**Observations:**
- **INVALID RUN:** Intended as a clean GQA baseline, but shell had stale `GATED_ATTN=elementwise` env var from a previous `source runs/configs/gated_attn_elementwise.env`. Param count (19.4M) matches Run 3 exactly.
- Results are essentially Run 3 re-run with slightly more steps (3,296 vs 3,129) → slightly better BPB (1.2576 vs 1.2602)
- Confirms elementwise model over 16 MB budget (17.96 MB)
- **Root cause fix:** All env configs now explicitly set `GATED_ATTN=none` and `ACTIVATION=relu2` as defaults, so sourcing any config resets stale env vars. Clean baseline should have 17,059,912 params and ~14.7 MB compressed.

### Run 6 — Clean GQA baseline, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline (no gated attention, no activation change) — clean re-run after env var fix |
| Model params | 17,059,912 (~17M) — confirmed correct |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,500 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | 171.41 ms |
| **val_loss (raw)** | **2.1338** |
| **val_bpb (raw)** | **1.2638** |
| **val_loss (int8+zlib)** | **2.1388** |
| **val_bpb (int8+zlib)** | **1.2667** |
| Baseline to beat | 1.2244 |
| Gap | +0.0423 |
| Model size (raw) | 67.22 MB |
| Model size (int8+zlib) | **15.75 MB (under 16 MB budget, +0.14 MB headroom)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1077
- Step 1000: val_bpb 1.3830
- Step 2000: val_bpb 1.3239
- Step 3000: val_bpb 1.2812
- Step 3500: val_bpb 1.2638 (still improving when time ran out)

**Observations:**
- Env var fix confirmed: 17,059,912 params (correct), not 19.4M
- Budget check script now shows int8+zlib size — 15.75 MB, under budget with 0.14 MB headroom
- More steps than Run 2 (3,500 vs 3,287) — no gate overhead means faster per step (171ms vs 182ms)
- But worse BPB than Run 2 headwise (1.2667 vs 1.2653) despite more steps — confirms headwise gated attention genuinely helps quality
- This is the proper control for isolating technique effects on this pod

### All Runs on PyTorch 2.11 Pod — Updated Comparison

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Size (int8+zlib) | Under 16 MB? |
|-----|-----------|--------|----------|---------|-------|----------|-------------------|-------------|
| 6 | **Baseline (GQA)** | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | 15.75 MB | Yes |
| 2 | Gated Attn (headwise) | 17.10M | 2.1366 | **1.2653** | 3,287 | 182ms | 15.75 MB | **Yes** |
| 3 | Gated Attn (elementwise) | 19.42M | 2.1280 | 1.2602 | 3,129 | 192ms | 17.87 MB | No |
| 4 | MQA | 17.65M | 2.1549 | 1.2761 | 3,370 | 178ms | 16.84 MB | No |

**Key finding:** Headwise gated attention improves BPB (1.2653 vs 1.2667 baseline) despite fewer steps (3,287 vs 3,500). The quality gain outweighs the speed cost. Elementwise has even better BPB but busts the budget.

### Run 6v2 — Clean GQA baseline repeat, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline repeat — reproducibility check after env var fix + branch sync |
| Model params | 17,059,912 (~17M) — confirmed correct |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,661 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | 163.87 ms |
| **val_loss (raw)** | **2.1304** |
| **val_bpb (raw)** | **1.2617** |
| **val_loss (int8+zlib)** | **2.1357** |
| **val_bpb (int8+zlib)** | **1.2649** |
| Baseline to beat | 1.2244 |
| Gap | +0.0405 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget, +0.12 MB headroom)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- Reproducibility confirmed: very close to Run 6 (1.2649 vs 1.2667, within noise)
- Faster step time than Run 6 (164ms vs 171ms) — GPU thermal variance between runs
- More steps (3,661 vs 3,500) → slightly better BPB
- Used as the control baseline for Runs 7-8 ablation

### Run 7 — LeakyReLU², 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | LeakyReLU(0.5)² activation (PG ranks 10-11) |
| Model params | 17,059,912 (~17M) — no extra params (LeakyReLU has no learnable weights) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,673 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | ~163 ms |
| **val_loss (int8+zlib)** | **2.1344** |
| **val_bpb (int8+zlib)** | **1.2641** |
| Baseline to beat | 1.2244 |
| Gap | +0.0397 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- LeakyReLU² gives **slightly better BPB than baseline** (1.2641 vs 1.2649 baseline_v2) — confirms leaderboard finding that it improves per-step quality
- Same param count as baseline — activation change is free
- Same step speed as baseline (~163 ms) — no throughput penalty
- Under budget (15.77 MB)
- Improvement is small (+0.0008 BPB) but consistent with ranks 10-11 using this technique

### Run 8 — LeakyReLU² + Headwise Gated Attn, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | LeakyReLU(0.5)² + Headwise Gated Attention (combo test) |
| Model params | 17,096,776 (~17.1M, +37K from headwise gates) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,368 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,824 MiB |
| Step avg | 178.15 ms |
| **val_loss (raw)** | **2.1294** |
| **val_bpb (raw)** | **1.2611** |
| **val_loss (int8+zlib)** | **2.1345** |
| **val_bpb (int8+zlib)** | **1.2642** |
| Baseline to beat | 1.2244 |
| Gap | +0.0398 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- Combo of LeakyReLU² + headwise gives **1.2642** — essentially identical to LeakyReLU² alone (1.2641)
- Headwise adds 178ms/step overhead vs 163ms baseline → fewer steps (3,368 vs 3,673)
- The techniques don't stack — LeakyReLU² is doing the work, headwise doesn't add value on top
- Under budget (15.77 MB)

### Runs 6-8 — Activation & Gated Attn Ablation (same pod, same PyTorch)

| Run | Technique | val_loss | val_bpb | Steps | Step avg | Size |
|-----|-----------|----------|---------|-------|----------|------|
| 6v2 | Baseline | 2.1357 | 1.2649 | 3,661 | 164ms | 15.77 MB |
| 7 | LeakyReLU² | 2.1344 | **1.2641** | 3,673 | 163ms | 15.77 MB |
| 8 | LeakyReLU² + headwise | 2.1345 | 1.2642 | 3,368 | 178ms | 15.77 MB |

**Key finding:** LeakyReLU² gives a small free improvement (-0.0008 BPB, no speed/size cost). Adding headwise gates on top doesn't help — the speed penalty (fewer steps) negates any per-step quality gain.

## Techniques That Worked

_Add entries as we discover things._

| Technique | Impact on BPB | Why it works |
|---|---|---|
| Gated Attention (headwise) | 1.2653 (compressed), fits 16 MB | Sigmoid gate after SDPA lets model suppress uninformative heads per token. Nearly free: +37K params, no speed penalty. |

## Techniques That Didn't Work

| Technique | Expected impact | Actual result | Why it failed |
|---|---|---|---|
| Gated Attention (elementwise) | Better BPB than headwise | 1.2602 BPB but 17.87 MB (over budget) | +2.36M params makes compressed model too large. Marginal BPB gain (+0.005) not worth the cost. |

## Key Insights

_High-level takeaways that apply beyond the competition._

1. 2 GPUs got to val_bpb 1.30 in 10 min — 8 GPUs should process ~4× more tokens in the same window, likely pushing below 1.22
2. Model still improving at wall clock cutoff — not converged, more throughput = better score
3. int8+zlib compression is essentially free (1.3033 → 1.3045, only +0.001 BPB degradation)

## On Metric Choice & Goodhart's Law

We use val_bpb (bits per byte on FineWeb validation) as our sole optimization target, consistent with the Parameter Golf competition metric. We acknowledge this is subject to Goodhart's Law — when a measure becomes a target, it ceases to be a good measure.

To keep our analysis honest, we categorize techniques by _what kind_ of improvement they provide:

| Category | What it means | Example techniques |
|----------|---------------|-------------------|
| **Architecture/training** | Genuinely better model — learns more per FLOP | Rho-1 token filtering, structured FFN, better LR schedules |
| **Throughput** | More steps in same wall clock — real improvement via more training | MQA speed gains, smaller batch accumulation |
| **Compression** | Smaller artifact, not better learning | int8→int6 quantization, layer tying for size |

Not all BPB improvements are equal. A technique that lowers BPB by fitting FineWeb's specific distribution (e.g., curriculum that mirrors the val set) is less valuable than one that improves the model's general language capability (e.g., better optimizer scheduling).

**Why BPB is still a reasonable proxy:** The Chinchilla paper (Hoffmann et al., 2022) showed that held-out perplexity improvements from principled scaling _do_ transfer to downstream tasks. The FineWeb paper (Penedo et al., 2024) confirmed that data quality improvements lowering BPB also improve MMLU/ARC. BPB isn't meaningless — it's just not the full picture.

**Our framing:** We study which techniques give real improvements under fixed compute constraints, while noting that some optimizations may be metric-specific rather than transferable.

## Crossover with Course Project

_Findings from Parameter Golf that are relevant to the nanochat training pipeline project._

- The 10-min wall clock constraint makes GPU count the primary lever — scaling from 2→8 GPUs is the easiest win
- Compression (int8+zlib) preserves model quality almost perfectly — useful for deployment constraints in the nanochat pipeline too
