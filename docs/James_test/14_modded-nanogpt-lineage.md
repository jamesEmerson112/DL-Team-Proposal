# Modded-NanoGPT → Parameter Golf: Lineage Investigation

**Author:** James Vo
**Date:** 2026-04-22
**Question:** Does Parameter Golf descend from modded-nanogpt? Is studying modded-nanogpt's 80 records worth our time?

## TL;DR

**Yes, PG is a direct fork of modded-nanogpt.** OpenAI explicitly credits it. But **no, you don't need to study modded-nanogpt's records** — PG already incorporates the key techniques, and the optimization target is fundamentally different (BPB under 16 MB vs. fastest time to 3.28 CE). Your time is better spent on PG runs.

---

## The Family Tree

```
nanoGPT (Karpathy, Jan 2023)
  │   Pure PyTorch GPT-2 reimplementation, educational
  │
  └─► modded-nanogpt (Keller Jordan, Jun 2024)
        │   Community speedrun: train GPT-2 to 3.28 CE as fast as possible
        │   80 records: 45 min → 1.4 min (32× speedup)
        │
        └─► Parameter Golf (OpenAI/Will DePue, Mar 2026)
              │   Forked and simplified for a different constraint:
              │   lowest BPB in 16 MB / 10 min on 8×H100
              │
              └─► Our experiments (Apr 2026)
                    Ablation study: isolate techniques, measure BPB deltas
```

## Evidence of Direct Lineage

| Source | Evidence |
|--------|----------|
| `parameter-golf/THIRD_PARTY_NOTICES.md` | "Portions of this repository were derived from modded-nanogpt, a project by Keller Jordan (Copyright 2024, MIT License)" |
| `parameter-golf/README.md` line 244 | "This repository adapts code from `modded-nanogpt`, see THIRD_PARTY_NOTICES.md" |
| `parameter-golf/train_gpt.py` line 100 | `# As borrowed from modded-nanogpt` (Muon optimizer section) |
| `parameter-golf/train_gpt.py` line 607 | `# relu^2 MLP from the original modded-nanogpt setup` |
| Git history | Initial commit `a15093a` (Mar 18, 2026) by Will DePue (OpenAI) already contains the full modded-nanogpt-derived codebase |

## What's Shared (Already in PG)

These techniques came from modded-nanogpt and are already in PG's `train_gpt.py`:

| Technique | Where in PG | Notes |
|-----------|-------------|-------|
| **Muon optimizer** | Lines 92-168 | Same Newton-Schulz coefficients (a=3.4445, b=-4.7750, c=2.0315) |
| **RoPE** | Lines 524-552 (`Rotary` class) | Standard rotary position encoding |
| **RMSNorm** | Lines 500-506 | Pre-norm residual structure |
| **ReLU²** | Lines 606-617 | Explicitly credited to modded-nanogpt |
| **GQA** | Lines 555-603 | 8 heads, 4 KV heads (configurable) |
| **QK-normalization** | CausalSelfAttention | With learnable q_gain |
| **CastedLinear** | Lines ~510 | fp32 storage, bf16 compute |
| **Flash Attention** | SDPA with flash_sdp backend | Via PyTorch's built-in |

**Conclusion:** The core algorithmic foundation is inherited. You don't need to go back to modded-nanogpt to get these.

## What PG Added (Not from modded-nanogpt)

PG's `train_gpt.py` extends the base with techniques for the 16 MB / BPB constraint:

| PG Addition | Purpose |
|-------------|---------|
| **U-Net skip connections** | Encoder-decoder style: first half stores skips, second half retrieves them in reverse. Learnable skip weights. |
| **Logit softcap** | `softcap * tanh(logits / softcap)` — stabilizes training |
| **Learnable residual mix** | Per-layer `resid_mix` blends current state with original embedding |
| **Per-layer scaling** | `attn_scale`, `mlp_scale` parameters per block |
| **int8 + zlib compression** | Post-training quantization for 16 MB artifact constraint |
| **Env-var config** | Every hyperparameter exposed via `os.environ.get()` |
| **1500-line limit** | Intentional simplification for accessibility |

## What's in Modded-NanoGPT But NOT in PG

These are advanced techniques from later speedrun records (records 30-80) that PG's baseline intentionally omits:

| Technique | Speedrun Record | Potential for PG? |
|-----------|----------------|-------------------|
| **Value embeddings** | #70 | Adds params — may not fit 16 MB |
| **Sliding window attention** | #40+ | Different constraint (PG has seq_len 1024, not a bottleneck) |
| **Multi-token prediction** | #53 | Could help BPB but adds complexity |
| **FP8 matmul** | #60+ | Throughput gain, but PG already fits in 10 min |
| **Polar Express orthogonalization** | Latest | Newer Muon variant, ~same FLOP cost |
| **Bigram hash embeddings** | #62 | Interesting for small vocab (1024) |
| **Fused Triton kernels** | #60+ | Speed optimization, less relevant for BPB |
| **Sparse gated attention** | #70+ | Could improve quality per FLOP |

## Why the Optimization Target Matters

| | Modded-NanoGPT Speedrun | Parameter Golf |
|---|---|---|
| **Metric** | Wall clock time to reach 3.28 CE | Lowest BPB (no target, just minimize) |
| **Constraint** | None on model size | 16 MB artifact cap |
| **Hardware** | 8×H100 (same) | 8×H100, 10-min wall clock cap |
| **What wins** | Faster kernels, communication overlap, curriculum | Better architecture per parameter, better training under fixed compute |

A technique that's valuable for speed (fused Triton kernels, FP8, communication overlap) may be irrelevant for BPB. Conversely, a technique that's valuable for BPB (structured FFN, layer tying, Rho-1 token selection) may be irrelevant for speed.

**This is why studying modded-nanogpt's records directly is low-priority** — most of the later records optimize for speed, not quality. The quality-improving techniques (Muon, RoPE, RMSNorm, ReLU², GQA) were already ported into PG at launch.

## What's Actually Worth Porting

From modded-nanogpt's later records, only a few techniques target model quality (BPB) rather than speed:

1. **Polar Express orthogonalization** — newer Muon variant, potentially better convergence. Low effort.
2. **Multi-token prediction** — predicting multiple next tokens could improve BPB. Medium effort.
3. **Bigram hash embeddings** — extra capacity for free at small vocab sizes. Medium effort.

Everything else is speed-focused and irrelevant to your BPB optimization.

## Recommendation

**Don't study modded-nanogpt's records.** Instead:

1. Use PG's `train_gpt.py` as-is (it already has the good stuff)
2. Focus on techniques from the NeurIPS paper survey (see `neurips-paper-survey.md`) — these target model quality, not speed
3. If you need a "related work" citation, one sentence is enough: "Parameter Golf adapts code from the modded-nanogpt speedrun community (Jordan et al., 2024), which drove GPT-2 training time from 45 minutes to under 1.5 minutes through algorithmic innovations including the Muon optimizer."

## Cross-Reference

This note extends `13_pg-vs-nanochat-architecture.md`, which compared PG with nanochat. The finding there ("both codebases descend from the modded-nanogpt speedruns") is now confirmed with direct evidence.

## Sources

- [modded-nanogpt repo](https://github.com/KellerJordan/modded-nanogpt)
- [Parameter Golf repo](https://github.com/openai/parameter-golf)
- [Muon optimizer blog](https://kellerjordan.github.io/posts/muon/)
- [Automated LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419)
- `parameter-golf/THIRD_PARTY_NOTICES.md`
- `parameter-golf/train_gpt.py` (lines 100, 607)
