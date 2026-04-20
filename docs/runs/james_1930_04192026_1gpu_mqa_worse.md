# Run Log — james_1930_04192026

**Author:** James Vo
**Date:** 2026-04-19 19:30
**Config:** explore_1gpu + NUM_KV_HEADS=1 (MQA)

## Change from Baseline

Switched from GQA (4 KV heads) to MQA (1 KV head) on single GPU.

## Results

| Item | Value |
|---|---|
| GPUs | 1x H100 SXM |
| Steps completed | 1,183 / 20,000 |
| val_loss | 2.2882 |
| **val_bpb** | **1.3552** |
| val_bpb (int8+zlib roundtrip) | **1.3565** |
| Baseline to beat | 1.2244 |
| Gap | +0.131 |
| Peak VRAM | 9,569 MiB |
| Step avg | 507 ms |
| Raw model size | 60.1 MB |
| Compressed (int8+zlib) | 12.2 MB |
| Under 16 MB budget? | Yes |
| Wall clock | 600,313 ms (hit cap) |

## Comparison vs 1-GPU GQA Baseline

| Metric | GQA (kv=4) | MQA (kv=1) | Delta |
|---|---|---|---|
| Steps in 10 min | 1,166 | 1,183 | +17 (negligible) |
| Step avg | ~515 ms | 507 ms | Barely faster |
| val_bpb | 1.3439 | 1.3552 | **+0.011 (WORSE)** |
| VRAM | 10,303 MiB | 9,569 MiB | -734 MiB |

## Finding

MQA does NOT help on single GPU. The speed advantage of fewer KV heads only materializes on multi-GPU (less inter-GPU sync). On 1 GPU there's no sync overhead to save, so you just lose model capacity with no speed gain.

**Recommendation:** Use GQA (default) for 1-GPU experiments. Reserve MQA for multi-GPU runs.
