# Run Log — james_1830_04192026

**Author:** James Vo
**Date:** 2026-04-19 18:30
**Config:** explore_2gpu + NUM_KV_HEADS=1 (MQA)

## Change from Baseline

Switched from GQA (4 KV heads) to MQA (1 KV head).

## Results

| Item | Value |
|---|---|
| GPUs | 2x H100 SXM |
| Steps completed | 3,767 / 20,000 |
| val_loss | 2.1502 |
| **val_bpb** | **1.2735** |
| val_bpb (int8+zlib roundtrip) | **1.2767** |
| Baseline to beat | 1.2244 |
| Gap | +0.052 |
| Peak VRAM | 9,540 MiB |
| Step avg | 159.27 ms |
| Raw model size | 60.1 MB |
| Compressed (int8+zlib) | 14.7 MB |
| Under 16 MB budget? | Yes |
| Wall clock | 599,966 ms (hit cap) |

## Comparison vs GQA Baseline

| Metric | GQA (kv=4) | MQA (kv=1) | Delta |
|---|---|---|---|
| Steps in 10 min | 1,770 | 3,767 | **+113%** |
| Step avg | 339 ms | 159 ms | **2.1x faster** |
| val_bpb | 1.3065 | 1.2735 | **-0.033 (better)** |
| VRAM | 10,303 MiB | 9,540 MiB | -763 MiB |

## Why It Worked

MQA reduces KV head count from 4 to 1, which:
1. Cuts attention computation — steps run 2.1x faster
2. Uses less VRAM (9.5 GB vs 10.3 GB)
3. More steps in same wall clock = more training = lower BPB
4. The quality tradeoff from sharing KV heads was outweighed by the extra training steps

## Notes

- First experiment that meaningfully closed the gap to baseline (1.2244)
- On 8 GPUs, this config would get ~15,000 steps in 10 min — likely beats baseline
- MQA is a keeper — use as new default for future experiments
