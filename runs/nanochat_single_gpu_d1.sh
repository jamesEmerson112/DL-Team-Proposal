#!/bin/bash

# Train a depth=1 nanochat model on a single GPU (RTX 5080 / 16 GB VRAM)
# This is the smallest possible model: d=1 → model_dim=64, n_heads=1 (~10K-50K params)
# Expect training to finish in under 10 minutes.
#
# Run as (from repo root):
#   bash runs/single_gpu_d1.sh
#
# Requirements:
#   - NVIDIA GPU with SM 7.0+ (RTX 5080 = SM 10.0, bf16 + Flash Attention supported)
#   - PyTorch 2.9.1+ (requires SM 7.0+; GTX 1070 Ti / SM 6.1 will NOT work)

set -euo pipefail

# cd into the nanochat submodule (works from repo root or from runs/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nanochat"

# -----------------------------------------------------------------------------
# Environment setup

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# wandb disabled by default — to enable:
#   1) wandb login
#   2) WANDB_MODE=online WANDB_RUN=d1_single bash runs/single_gpu_d1.sh
export WANDB_MODE="${WANDB_MODE:-disabled}"
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Data + tokenizer (skip if already done)

SHARD_DIR="$NANOCHAT_BASE_DIR/data"
TOK_FILE="$NANOCHAT_BASE_DIR/tokenizer.model"

if [ ! -d "$SHARD_DIR" ] || [ -z "$(ls -A "$SHARD_DIR" 2>/dev/null)" ]; then
    echo "==> Downloading dataset (8 shards, ~800 MB)..."
    python -m nanochat.dataset -n 8
else
    echo "==> Dataset shards already present, skipping download."
fi

if [ ! -f "$TOK_FILE" ]; then
    echo "==> Training tokenizer..."
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
else
    echo "==> Tokenizer already trained, skipping."
fi

# -----------------------------------------------------------------------------
# Train d=1 on 1 GPU
# Using torchrun for DDP compatibility even on a single GPU.
# d=1 with head_dim=64 → model_dim=64, n_heads=1

echo ""
echo "============================================"
echo "  Training depth=1 model on 1 GPU"
echo "  (RTX 5080 / SM 10.0 — bf16 auto-detected)"
echo "============================================"
echo ""

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=1 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --num-iterations=5000 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --save-every=-1 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Print model size + Parameter Golf budget check

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
python -c "
import torch, glob, os

base = os.environ['NANOCHAT_BASE_DIR']
# d=1, head_dim=64 → model_dim=64, n_heads=1

model_dim = 1 * 64  # depth * head_dim
vocab_size = 32768   # 2^15 default

# Approximate: embedding + lm_head + transformer layers
embed_params = vocab_size * model_dim * 2  # token_emb + lm_head (weight-tied usually)
# Each transformer layer: ~12 * model_dim^2 (rough estimate for attention + FFN)
layer_params = 1 * 12 * model_dim * model_dim
total_approx = embed_params + layer_params

bf16_bytes = total_approx * 2
bf16_mb = bf16_bytes / (1024 * 1024)

print(f'  Estimated params:    ~{total_approx:,}')
print(f'  Model size (bf16):   ~{bf16_mb:.2f} MB')
print(f'  Parameter Golf cap:   16.00 MB')
print(f'  Under budget:         {\"YES\" if bf16_mb <= 16.0 else \"NO\"} ({16.0 - bf16_mb:.2f} MB headroom)')
print()
print('  Check training stdout above for exact val_bpb and param count.')
"

echo ""
echo "Done. Check stdout above for val_bpb and training loss."
