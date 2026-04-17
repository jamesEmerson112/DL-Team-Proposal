#!/bin/bash

# Run the OpenAI Parameter Golf baseline (FineWeb sp1024, 1024 vocab)
# This is SEPARATE from nanochat — it uses openai/parameter-golf's train_gpt.py.
#
# Run as (from repo root):
#   bash runs/parameter_golf_baseline.sh
#
# For 8xH100 full run:
#   NGPUS=8 bash runs/parameter_golf_baseline.sh
#
# With fewer data shards (faster, for testing):
#   TRAIN_SHARDS=4 bash runs/parameter_golf_baseline.sh
#
# Competition: https://openai.com/index/parameter-golf/
# Repo:        https://github.com/openai/parameter-golf
# Deadline:    April 30, 2026
# Baseline:    1.2244 BPB (9 layers, 512 dims, 1024 vocab)

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via env vars)

NGPUS="${NGPUS:-1}"                          # GPUs to use (1 = smoke test, 8 = full run)
RUN_ID="${RUN_ID:-baseline_sp1024}"          # wandb / logging run name
TRAIN_SHARDS="${TRAIN_SHARDS:-}"            # empty = full dataset (80 shards)

# -----------------------------------------------------------------------------
# Locate parameter-golf repo (sibling to DL-Team-Proposal)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/../parameter-golf"

# Clone if not present
if [ ! -d "$PG_DIR" ]; then
    echo "==> Cloning openai/parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$PG_DIR"
else
    echo "==> parameter-golf repo already present at $PG_DIR"
fi

cd "$PG_DIR"
echo "==> Working in $(pwd)"

# -----------------------------------------------------------------------------
# Install dependencies

if [ -f "requirements.txt" ]; then
    echo "==> Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "==> Installing package..."
    pip install -e .
else
    echo "==> No requirements file found, installing common deps..."
    pip install torch sentencepiece
fi

# -----------------------------------------------------------------------------
# Download dataset (skip if already present)

DATA_DIR="./data/datasets/fineweb10B_sp1024"
TOKENIZER_FILE="./data/tokenizers/fineweb_1024_bpe.model"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "==> Downloading FineWeb sp1024 dataset..."
    DOWNLOAD_CMD="python3 data/cached_challenge_fineweb.py --variant sp1024"
    if [ -n "$TRAIN_SHARDS" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --train-shards $TRAIN_SHARDS"
    fi
    $DOWNLOAD_CMD
else
    echo "==> Dataset already present, skipping download."
fi

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "WARNING: Tokenizer not found at $TOKENIZER_FILE"
    echo "         The dataset download should have created it."
fi

# -----------------------------------------------------------------------------
# Train

echo ""
echo "============================================"
echo "  Parameter Golf — Training Baseline"
echo "  GPUs: $NGPUS | Run: $RUN_ID"
echo "  Vocab: 1024 | Dataset: FineWeb sp1024"
echo "============================================"
echo ""

export RUN_ID
export DATA_PATH="$DATA_DIR/"
export TOKENIZER_PATH="$TOKENIZER_FILE"
export VOCAB_SIZE=1024

torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

# -----------------------------------------------------------------------------
# Results + artifact size check

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
echo ""

# Check artifact size (code + weights must fit in 16 MB = 16,000,000 bytes)
python3 -c "
import os, glob

budget = 16_000_000  # 16 MB in bytes

# Find model weights (common patterns)
weight_files = glob.glob('*.pt') + glob.glob('*.pth') + glob.glob('*.bin') + glob.glob('model.*')
code_files = glob.glob('train_gpt*.py')

total = 0
print('  Artifact components:')
for f in sorted(set(weight_files + code_files)):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        total += size
        print(f'    {f}: {size:,} bytes ({size/1e6:.2f} MB)')

print(f'')
print(f'  Total artifact:      {total:,} bytes ({total/1e6:.2f} MB)')
print(f'  Budget:              {budget:,} bytes (16.00 MB)')
if total > 0:
    print(f'  Under budget:        {\"YES\" if total <= budget else \"NO\"} ({(budget - total)/1e6:.2f} MB headroom)')
else:
    print(f'  (no weight files found — check training output)')
print()
print(f'  Baseline to beat:    1.2244 BPB')
print(f'  Check training stdout above for your val_bpb.')
"

echo ""
echo "Done."
