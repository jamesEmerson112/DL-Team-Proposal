# Run: v4_ts_embed_seed42_2gpu

**Pod:** 2×H100 80GB SXM
**Seed:** 42
**Data:** Kevin Clark's vanilla SP8192 (kevclark/parameter-golf), 80 train shards
**Script:** train_v4.py (train_v1.py + Universal-Transformer timestep embed on loop iterations)
**Config:** v1_base.env + v4_ts_embed.env (TS_EMBED_ENABLED=1, TTT_ENABLED=0)

**Note:** TTT was disabled for faster iteration (TTT_ENABLED=0). Headline is pre-TTT and post-quant (no TTT).

## What v4 adds

Rank 4's Loop4-5 × 2 reuses the same layers 4 and 5 in pass 1 and pass 2.
Without a per-iteration signal, the shared weights can't distinguish iterations
— Universal Transformer's timestep embed trick addresses exactly this.

v4 adds a learned `(num_loops+1, model_dim)` tensor (= 3×512 = 1,536 fp32 params,
~6 KB raw), zero-init. At each visit to a layer inside [loop_start, loop_end],
the corresponding row is added to x before the block's attn_norm. Precomputed
visit indices make it compile-friendly.

## Headline

| Metric | v1 (anchor) | v4 (ts_embed) | Δ vs v1 |
|---|---|---|---|
| Params | 35,988,634 | 35,990,170 | **+1,536** (exactly 3×512, matches spec ✓) |
| Steps | 1,379 | 1,359 | -20 (tiny slowdown from extra add ops) |
| Pre-TTT val_bpb | 1.14564632 | **1.14671062** | **+0.00106 (noise-level)** |
| Quantized val_bpb | 1.15192376 | **1.15303547** | +0.00111 |
| Post-TTT val_bpb | 1.13735643 | (disabled) | n/a |
| Artifact | 15,991,932 B | 15,994,470 B | +2,538 B |
| Peak VRAM | 35,942 MiB | 36,149 MiB | +207 MiB |

## Verdict

**Tied with v1 (within noise).** The ~0.001 delta is ~1.5 σ on rank 4's 5-seed
stdev of 0.00068 — can't distinguish "mildly negative" from "zero" at 1 seed.

Likely cause for the no-show: zero-init means ts_embed has to actively learn to
move away from zero. On our truncated 2×H100 run, looping only activates for
~900 steps (frac ≥ 0.35), only 2 of 11 layers are in the loop, and the visit
counter only takes 3 values. Not a lot of gradient signal to shape a new
parameter.

**If you want to pursue this further**, options in order of likelihood to help:
1. **Small normal init** (std=0.005) instead of zero — gives the model a
   non-degenerate starting point. Risk: might destabilize for first few steps.
2. **Loop more layers** (e.g., Loop3-5 × 2 or Loop3-6 × 2) — more of the model
   uses the signal, more gradient volume.
3. **More loop iterations** (NUM_LOOPS=3 or 4) — paper territory; adds cost.

At 1 seed on 2×H100 budget, I'd say this axis isn't productive to pursue further.
Norm axis (v2, v3) also regressed. Suggests rank 4's architecture is well-tuned
and wins on this stack are more likely to come from data/optimizer/quantization
axes rather than architectural adds.

## Raw grep from log

```
  ts_embed_enabled: True
model_params:35990170
stopping_early: wallclock_cap train_time: 596314ms step: 1359/20000
peak memory allocated: 36149 MiB reserved: 40336 MiB
diagnostic pre-quantization post-ema val_loss:2.96218064 val_bpb:1.14671062 eval_time:14766ms
Total submission size quantized+brotli: 15994470 bytes
diagnostic quantized val_loss:2.97851899 val_bpb:1.15303547 eval_time:77386ms
```
