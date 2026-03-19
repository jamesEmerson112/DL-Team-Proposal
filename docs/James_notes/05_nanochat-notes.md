# nanochat — Research Notes

**Author:** Andrej Karpathy
**Repo:** github.com/karpathy/nanochat
**License:** MIT
**Released:** October 13, 2025
**Status:** Active — official successor to nanoGPT
**Tagline:** "The best ChatGPT that $100 can buy."

## Overview

NanoChat is a full-stack, end-to-end pipeline for building a complete ChatGPT-style model — from tokenizer training through pretraining, fine-tuning, reinforcement learning, evaluation, and interactive chat deployment. ~3,500 lines across ~30 files.

## Design Philosophy

- Minimal, readable, hackable, maximally-forkable "strong baseline"
- Single `--depth` parameter ("Complexity Dial") auto-configures ALL hyperparameters via scaling laws
- Demonstrates you can train a GPT-2-capability chat model for ~$48–$100

## Architecture Improvements Over nanoGPT

| Feature | nanoGPT (GPT-2) | nanochat |
|---|---|---|
| Position encoding | Learned absolute | **Rotary Position Embeddings (RoPE)** |
| Normalization | LayerNorm | **RMSNorm** |
| Activation | GELU | **ReLU-squared** |
| Attention | Standard multi-head | **Multi-Query Attention (MQA)** with QK normalization |
| Embeddings | Tied (wte = lm_head) | **Untied** (separate wte and lm_head) |
| Attention impl | Flash Attention (optional) | **Flash Attention 3** with SDPA fallback |
| Special features | None | Value embeddings, per-layer residual scalars, sliding window attention |
| Optimizer | AdamW | **Muon** (weight matrices) + **AdamW** (embeddings) |
| Precision | PyTorch autocast | Explicit dtype (bf16 on A100/H100, fp32 older) |

**Standard model (depth-20):** ~560M params, 1,280 hidden dims, 10 attention heads (dim 128 each), trained on 11.2B tokens.

## Training Pipeline (4 Stages)

### Stage 1 — Tokenization (~30 min)
- Custom **Rust-based BPE tokenizer** trained from scratch
- 32,768 or 65,536 token vocabulary (configurable)
- ~4.8x compression ratio on FineWeb text
- Includes special tokens for conversation formatting and tool use

### Stage 2 — Pretraining (~2.5–3 hours)
- **Dataset:** FineWeb-EDU (or NVIDIA ClimbMix for speedrun entries)
- Distributed via `torchrun` on 8 GPUs
- BOS-aligned data loading (every sequence starts at document boundary)
- Targets Chinchilla-optimal token:parameter ratios (~20:1)

### Stage 3 — Mid-training / SFT (~8–30 min)
- Adapts base model for conversations using task mixture:
  - **SmolTalk:** 460K conversation examples
  - **MMLU auxiliary-train:** 100K multiple-choice questions
  - **GSM8K:** 8K grade-school math problems
  - **SpellingBee:** Additional task data
- Teaches tool use via `<|python_start|>...<|python_end|>` special tokens

### Stage 4 — Reinforcement Learning (optional, ~1 hour)
- **GRPO** (Group Relative Policy Optimization) — simplified RLHF variant
- Applied to GSM8K math problems
- Still experimental

## Datasets

| Stage | Dataset | Details |
|---|---|---|
| Tokenizer | ClimbMix shards | 8 parquet shards (~800MB) |
| Pretraining | FineWeb-EDU-100B | 1,822 shards (~100MB each); depth-20 uses 240 shards |
| Pretraining (alt) | NVIDIA ClimbMix-400B | Used for leaderboard speedruns |
| Mid-training | SmolTalk + MMLU + GSM8K | 568K rows combined |

## Benchmarks — DCLM CORE Score

22-task ensemble spanning reasoning, knowledge, math, code, and common sense:

**Tasks include:** ARC (Easy + Challenge), HellaSwag, MMLU, TriviaQA, GSM8K, HumanEval, and others.

### Performance at Different Training Durations

| Duration | Cost | Achievement |
|---|---|---|
| ~2 hours | ~$48 | Matches GPT-2 CORE baseline (0.256) |
| 4 hours | ~$100 | Basic conversational ability, CORE ~0.269 |
| 12 hours | ~$300 | Surpasses GPT-2 on CORE |
| 24 hours | ~$600 | 40% MMLU, 70% ARC-Easy |

### Depth-20 Model Benchmarks

| Benchmark | Score |
|---|---|
| CORE | 0.22 |
| MMLU | 31.5% |
| ARC-Easy | 38.8% |
| GSM8K | 4.6% |
| HumanEval | 8.5% |

## Inference Engine

- KV cache optimized for Flash Attention 3
- Efficient prefill/decode separation
- Built-in Python interpreter sandbox for tool use
- Streaming token-by-token generation

## Chat Interface

- **CLI:** `python -m scripts.chat_cli`
- **Web UI:** `python -m scripts.chat_web` — FastAPI on port 8000 with ChatGPT-style HTML frontend
- Multi-GPU via `WorkerPool` for data-parallel inference

## Hardware Requirements

| Tier | Setup | Notes |
|---|---|---|
| Recommended | 8xH100 | ~$24/hr from Lambda GPU Cloud |
| Minimum | Single GPU | Gradient accumulation (8x slower) |
| CPU/Apple Silicon | `runs/runcpu.sh` | Dramatically reduced model size |

VRAM tuning: `--device_batch_size` adjustable (32 → 16 → 8 → 4 → 2 → 1) for GPUs with less VRAM.

## Key Files

```
nanochat/
  nanochat/
    gpt.py           # Transformer model
    engine.py         # KV cache inference engine
    tokenizer.py      # BPE tokenizer wrapper
    ui.html           # ChatGPT-style web frontend
  scripts/
    chat_web.py       # FastAPI web server
    chat_cli.py       # Terminal chat interface
    chat_sft.py       # Supervised fine-tuning
    chat_rl.py        # Reinforcement learning (GRPO)
  base_train.py       # Pretraining script
  runs/
    speedrun.sh       # Full pipeline (tokenize → pretrain → SFT → deploy)
    runcpu.sh         # CPU/Apple Silicon config
  tasks/              # Evaluation tasks (ARC, GSM8K, MMLU, etc.)
```

## Community Variants

- **nanogpt-community/nanochat** — open-source self-hostable chat client
- **nanochat-mlx** — Apple Silicon optimized port
