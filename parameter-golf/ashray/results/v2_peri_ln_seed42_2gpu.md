# Run: v2_peri_ln_seed42_2gpu

**Pod:** 2×H100 80GB SXM
**Date:** Thu Apr 30 17:38:05 UTC 2026
**Seed:** 42
**Data:** Kevin Clark's vanilla SP8192 (kevclark/parameter-golf), 80 train shards
**Script:** train_v2.py (train_v1.py + Peri-LN flag)
**Config:** v1_base.env + v2_peri_ln.env (PERI_LN_ENABLED=1)

**Note:** TTT phase was killed after observing the large pre-TTT regression vs v1.
Headline is the pre-TTT / quantized-pre-TTT values.

## Headline

| Metric | Value | v1 baseline | Δ vs v1 |
|---|---|---|---|
| Steps | 1322/20000 (wallclock cap at 596s) | 1379 | −57 |
| Pre-TTT val_bpb | 1.18423421 | 1.14564632 | **+0.03859 (worse)** |
| Quantized val_bpb | 1.19235742 | 1.15192376 | **+0.04043 (worse)** |
| Post-TTT val_bpb | (TTT killed) | 1.13735643 | n/a |
| Artifact | 15,982,109 bytes (15.98 MB / 16 MB) | 15,991,932 | −9,823 B |
| Params | 35,988,634 | 35,988,634 | same (Peri-LN is zero-param) |
| Peak VRAM | 35,950 MiB | 35,942 MiB | ≈tied |

## Conclusion

Peri-LN regressed substantially on rank 4 + vanilla SP8192.
The delta (+0.039) is ~50× rank 4's 5-seed stdev of 0.00068 — not noise.
**Decision: drop Peri-LN from the stack.** Do not include in v3+.

## Raw grep from log

```
  peri_ln_enabled: True
model_params:35988634
stopping_early: wallclock_cap train_time: 596446ms step: 1322/20000
peak memory allocated: 35950 MiB reserved: 41116 MiB
diagnostic pre-quantization post-ema val_loss:3.05911151 val_bpb:1.18423421 eval_time:13979ms
Total submission size quantized+brotli: 15982109 bytes
diagnostic quantized val_loss:3.08009537 val_bpb:1.19235742 eval_time:16730ms
```
