# Parameter Golf — Findings & Insights

> What we learned from participating. Insights that could feed back into the course project.

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
| **val_bpb** | **1.3045** |
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

## Techniques That Worked

_Add entries as we discover things._

| Technique | Impact on BPB | Why it works |
|---|---|---|
| TBD | TBD | TBD |

## Techniques That Didn't Work

| Technique | Expected impact | Actual result | Why it failed |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

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
