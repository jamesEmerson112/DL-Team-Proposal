# Run: v1_baseline_seed42_2gpu

**Pod:** 2×H100 80GB SXM
**Date:** Thu Apr 30 16:22:42 UTC 2026
**Seed:** 42
**Data:** Kevin Clark's vanilla SP8192 (kevclark/parameter-golf), 80 train shards
**Script:** train_v1.py (rank 4 PR #1769, unmodified)
**Config:** v1_base.env (CASEOPS_ENABLED=0, MIN_LR=0.10)

## Headline

| Metric | Value |
|---|---|
| Steps | 1379/20000 (wallclock cap at 596s) |
| Pre-TTT val_bpb | 1.14564632 |
| Quantized val_bpb | 1.15192376 |
| **Post-TTT val_bpb** | **1.13735643** |
| Artifact | 15,991,932 bytes (15.99 MB / 16 MB) |
| Eval time | 859s (over 10min cap, expected on 2GPU) |

## Raw grep from log

```
model_params:35988634
stopping_early: wallclock_cap train_time: 596377ms step: 1379/20000
peak memory allocated: 35942 MiB reserved: 40070 MiB
diagnostic pre-quantization post-ema val_loss:2.95943135 val_bpb:1.14564632 eval_time:13725ms
Total submission size quantized+brotli: 15991932 bytes
diagnostic quantized val_loss:2.97564723 val_bpb:1.15192376 eval_time:74535ms
quantized_ttt_phased val_loss:2.93790979 val_bpb:1.13735643 eval_time:859395ms
total_eval_time:859.4s
```
