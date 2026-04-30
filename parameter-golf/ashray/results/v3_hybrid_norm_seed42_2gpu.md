# Run: v3_hybrid_norm_seed42_2gpu

**Pod:** 2×H100 80GB SXM
**Date:** Thu Apr 30 17:53:46 UTC 2026
**Seed:** 42
**Data:** Kevin Clark's vanilla SP8192 (kevclark/parameter-golf), 80 train shards
**Script:** train_v3.py (train_v1.py + V-norm + Post-Norm FFN flags)
**Config:** v1_base.env + v3_hybrid_norm.env (V_NORM_ENABLED=1, POST_NORM_FFN_ENABLED=1)

**Note:** TTT phase was killed / did not complete. Headline is the pre-TTT value.

## Headline

| Metric | Value | v1 baseline | Δ vs v1 |
|---|---|---|---|
| Steps | 1334/20000 (wallclock cap at 596s) | 1379 | −45 |
| Pre-TTT val_bpb | 1.15640560 | 1.14564632 | **+0.01076 (worse)** |
| Quantized val_bpb | (not captured) | 1.15192376 | n/a |
| Post-TTT val_bpb | (TTT killed) | 1.13735643 | n/a |
| Params | 35,988,634 | 35,988,634 | same (zero-param) |
| Peak VRAM | 35,541 MiB | 35,942 MiB | −401 MiB |

## Comparison across norm-axis experiments

| Approach | Pre-TTT val_bpb | Δ vs v1 | verdict |
|---|---|---|---|
| v1 (no extra norm) | 1.14565 | — | anchor |
| v2 Peri-LN (norm sublayer outputs) | 1.18423 | +0.03859 | **big regression** |
| v3 HybridNorm (V-norm + Post-Norm FFN) | 1.15641 | +0.01076 | smaller regression |

## Conclusion

HybridNorm regressed less than Peri-LN but still regressed. Both norm-axis
experiments hurt on rank 4 + vanilla SP8192 at 1-seed 2×H100. Rank 4's stack
is already heavily normalized (Q/K-norm in attention, ln_scale_factor per
layer, resid_mix, attn_scale/mlp_scale); adding more normalization seems
to cancel or conflict with the learned residual magnitudes.

**Decision: drop HybridNorm from the stack.** Norm axis appears closed
for our setup. Next experiments should target different axes (e.g.,
attention computation, optimizer, quantization).

## Raw grep from log

```
  post_norm_ffn_enabled: True
  v_norm_enabled: True
model_params:35988634
stopping_early: wallclock_cap train_time: 596099ms step: 1334/20000
peak memory allocated: 35541 MiB reserved: 39732 MiB
diagnostic pre-quantization post-ema val_loss:2.98722471 val_bpb:1.15640560 eval_time:17876ms
```
