# Run Log — james_1900_04192026

**Author:** James Vo
**Date:** 2026-04-19 19:00
**Config:** explore_1gpu, GQA baseline (default)

## Results

| Item | Value |
|---|---|
| GPUs | 1x H100 SXM |
| Steps completed | 1,166 / 20,000 |
| val_loss | 2.2691 |
| **val_bpb** | **1.3439** |
| val_bpb (int8+zlib roundtrip) | **1.3439** |
| Baseline to beat | 1.2244 |
| Gap | +0.120 |
| Peak VRAM | 10,303 MiB |
| Step avg | ~515 ms |
| Raw model size | 67.2 MB |
| Compressed (int8+zlib) | 12.9 MB |
| Under 16 MB budget? | Yes |
| Wall clock | 600,478 ms (hit cap) |

## Purpose

Establish the 1-GPU GQA baseline so teammates can compare their experiments on single-GPU pods. This is the number to beat for 1-GPU experiments.

## Notes

- 1-GPU gets ~1,166 steps in 10 min vs ~1,770 on 2-GPU (34% fewer)
- GQA is the correct default for single-GPU — MQA does not help (see Run james_1930)
