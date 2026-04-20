# Run Log — james_1800_04192026

**Author:** James Vo
**Date:** 2026-04-19 18:00
**Config:** explore_2gpu (2xH100 SXM, 10-min wall clock)

## Results

| Item | Value |
|---|---|
| GPUs | 2x H100 SXM |
| Steps completed | 1,770 / 20,000 |
| val_loss | 2.2059 |
| **val_bpb** | **1.3065** |
| val_bpb (int8+zlib roundtrip) | **1.3076** |
| Baseline to beat | 1.2244 |
| Gap | +0.083 |
| Model params | 17,059,912 (~17M) |
| Peak VRAM | 10,303 MiB |
| Step avg | 339.04 ms |
| Raw model size | 67.2 MB |
| Compressed (int8+zlib) | 14.6 MB |
| Under 16 MB budget? | Yes |
| Wall clock | 600,100 ms (hit cap) |

## Training Curve

| Step | Metric | Value |
|---|---|---|
| 0 | val_bpb | 4.1077 |
| 1 | train_loss | 6.9357 |
| 5 | train_loss | 6.6521 |
| 10 | train_loss | 5.9958 |
| 1,770 | val_bpb | 1.3065 |

## Notes

- Reproducible baseline on 2xH100 — consistent with Run 1 (Apr 16, val_bpb 1.3045)
- Model still improving at wall clock cutoff
- PyTorch upgraded from 2.4.1 to 2.11.0 to fix `enable_gqa` error in `scaled_dot_product_attention`
- Compression nearly lossless: 1.3065 → 1.3076 (+0.001 BPB)
