# nanoGPT — Research Notes

**Author:** Andrej Karpathy
**Repo:** github.com/karpathy/nanoGPT
**License:** MIT
**Status:** Deprecated (Nov 2025) — superseded by nanochat

## Overview

NanoGPT is a minimalist repository for training and fine-tuning medium-sized GPT models. The entire core implementation fits in ~600 lines of Python across two files. It's a rewrite of Karpathy's earlier `minGPT` that "prioritizes teeth over education" — practical and performant rather than purely pedagogical.

## Design Philosophy

- ~300 lines for training loop (`train.py`), ~300 lines for model (`model.py`)
- No heavy abstractions, no factory patterns
- Easy to fork, modify, and experiment with
- Leverages PyTorch 2.0: `torch.compile()`, Flash Attention, `np.memmap` data loading

## Architecture

Standard **decoder-only Transformer** (GPT-2 architecture):

| Component | Description |
|---|---|
| `GPTConfig` | 7 params: block_size (1024), vocab_size (50304), n_layer (12), n_head (12), n_embd (768), dropout (0.0), bias (True) |
| `CausalSelfAttention` | Masked multi-head self-attention with optional Flash Attention |
| `MLP` | Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) |
| `Block` | LayerNorm → Attention → Residual → LayerNorm → MLP → Residual |
| `GPT` | Token Emb + Position Emb → N Blocks → LayerNorm → Linear Head |

### Supported GPT-2 Variants

| Model | Layers | Heads | Embed Dim | Params |
|---|---|---|---|---|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1.5B |

Can load pre-trained GPT-2 weights from OpenAI via HuggingFace for fine-tuning.

## Training

- Cosine learning rate decay with linear warmup
- Gradient accumulation for large effective batch sizes
- Multi-GPU via PyTorch DDP, multi-node support
- Periodic checkpointing (saves on val loss improvement)
- Optional Weights & Biases logging
- `torch.compile()` optimization

**Benchmark:** Reproduces GPT-2 (124M) on OpenWebText in ~4 days on 8xA100 40GB, achieving val loss ~3.12.

## Tokenization

Uses GPT-2 BPE tokenizer via `tiktoken`. Tokens stored as `np.uint16` (max GPT-2 token ID is 50,256). Tokenization parallelized across CPU cores.

## Datasets

| Dataset | Use Case | Details |
|---|---|---|
| OpenWebText | Full GPT-2 reproduction | ~8M docs from Reddit-linked pages (3+ upvotes) |
| Shakespeare | Quick start | ~1MB, character-level, trains in ~3 min on A100 |

Each dataset has a `prepare.py` in `data/` that downloads, tokenizes, and saves `train.bin` / `val.bin`.

## Key Files

```
nanoGPT/
  train.py          # Training loop (~300 lines)
  model.py          # GPT model definition (~300 lines)
  sample.py         # Inference / text generation
  configurator.py   # Config override system
  config/           # Presets (train_shakespeare_char.py, etc.)
  data/
    openwebtext/prepare.py
    shakespeare/prepare.py
    shakespeare_char/prepare.py
```

## Usage

```bash
# Quick start (Shakespeare, character-level)
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char

# Full GPT-2 reproduction
python data/openwebtext/prepare.py
python train.py config/train_gpt2.py

# Fine-tuning pre-trained GPT-2
python train.py config/finetune_shakespeare.py
```

## Related

- **build-nanogpt** (github.com/karpathy/build-nanogpt) — companion video+code lecture rebuilding nanoGPT from scratch. YouTube: "Let's reproduce GPT-2 (124M)". Training now takes ~1 hour / ~$10 on modern cloud GPUs.
