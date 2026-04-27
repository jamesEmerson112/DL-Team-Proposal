# Parameter Golf vs nanochat — Architecture Comparison

**Author:** James Vo
**Date:** 2026-04-19
**Source:** Side-by-side code review of `parameter-golf/train_gpt.py` and nanochat's `gpt.py` / `base_train.py`

## Key Finding

Both codebases descend from the **modded-nanogpt speedruns** community. Most modern training techniques are already shared — there is very little to "port" from nanochat into PG.

## Architecture Comparison

| Component | PG `train_gpt.py` | nanochat `gpt.py` | Same? |
|---|---|---|---|
| **Position encoding** | RoPE (lines 524-552) | RoPE | Already same |
| **Normalization** | RMSNorm (lines 500-506) | RMSNorm | Already same |
| **Activation** | ReLU² (lines 606-617) | ReLU² | Already same |
| **Optimizer** | Muon (matrices) + Adam (embeds) (lines 92-168) | Muon (matrices) + AdamW (embeds) | Already same |
| **Attention** | GQA (8 heads, 4 kv_heads) | MQA (n heads, 1 kv_head) | Different |
| **QK normalization** | Yes (rms_norm + learnable q_gain) | Yes (QK norm) | Already same |
| **Embeddings** | Tied (default) | Untied (separate wte + lm_head) | Different |
| **Skip connections** | U-Net style (encoder/decoder halves) | Standard residual | PG is more advanced |
| **Residual mixing** | Learned mix of x and x0 | Per-layer residual scalars | Different approach |
| **Logit softcap** | Yes (tanh capping, default 30.0) | No | PG has extra |
| **Flash Attention** | SDPA with flash_sdp | Flash Attention 3 | Similar |
| **Weight precision** | CastedLinear (fp32 store, bf16 compute) | bf16 / explicit dtype | PG is more careful |
| **Compression** | int8+zlib (built-in) | N/A | PG only |
| **Vocab** | 1024 (tiny) | 32,768-65,536 | Different constraints |
| **Seq length** | 1024 | 2048 | Different |
| **Auto-config** | Manual (all knobs exposed via env vars) | `--depth` auto-derives everything | Different philosophy |
| **SFT / RLHF** | N/A (pretrain only) | Full pipeline | N/A for PG |

## What This Means for Our Experiments

Since most techniques are already shared, the levers we can actually pull are:

1. **Gated attention** — from NeurIPS 2025 Best Paper (not in either codebase yet)
2. **Tied vs untied embeddings** — PG ties by default; nanochat unties
3. **Vocab size** — try 4096 or 8192 instead of 1024
4. **Hyperparameter tuning** — learning rates, warmup/warmdown, batch size, qk_gain_init
5. **Data strategies** — shard selection, ordering, curriculum
6. **Architecture search** — layer count, width, MLP ratio, head count within 16 MB budget

## Current Scores

| Entry | BPB |
|---|---|
| Our Run 1 (2 GPU, Apr 16) | 1.3045 |
| PG baseline | 1.2244 |
| Current SOTA (28 submissions) | 1.0810 |
