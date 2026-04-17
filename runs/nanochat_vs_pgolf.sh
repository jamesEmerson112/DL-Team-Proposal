#!/usr/bin/env bash
# nanochat_vs_pgolf.sh — Apples-to-apples BPB comparison
# Same dataset (FineWeb sp1024), same metric (BPB), two engines:
#   1. Parameter Golf's train_gpt.py
#   2. nanochat's base_train.py (with pretokenized data support)
#
# Usage:
#   bash runs/nanochat_vs_pgolf.sh          # full run
#   TRAIN_SHARDS=2 bash runs/nanochat_vs_pgolf.sh  # quick smoke test
#
# Prerequisites:
#   - CUDA GPU(s) available
#   - nanochat installed: cd nanochat && uv sync --extra gpu --extra pgolf
#   - Python 3.10+

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parameter Golf repo (cloned alongside this repo by default)
PG_DIR="${PG_DIR:-$REPO_ROOT/parameter-golf}"
PG_DATA_DIR="${PG_DATA_DIR:-$PG_DIR/data/fineweb_sp1024}"
PG_TOKENIZER="${PG_TOKENIZER:-$PG_DIR/data/fineweb_sp1024/tok1024.model}"

# GPU config
NGPUS="${NGPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

# Training config — match Parameter Golf defaults for fair comparison
DEPTH="${DEPTH:-9}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

# Optional: limit training shards for smoke testing
TRAIN_SHARDS="${TRAIN_SHARDS:-}"

echo "============================================================"
echo " nanochat vs Parameter Golf — BPB Comparison"
echo "============================================================"
echo "GPUs:            $NGPUS"
echo "Depth:           $DEPTH"
echo "Iterations:      $NUM_ITERATIONS"
echo "Eval every:      $EVAL_EVERY"
echo "Device batch:    $DEVICE_BATCH_SIZE"
echo "Max seq len:     $MAX_SEQ_LEN"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Clone Parameter Golf and download data (skip if present)
# ---------------------------------------------------------------------------
if [ ! -d "$PG_DIR" ]; then
    echo "[1/4] Cloning openai/parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$PG_DIR"
else
    echo "[1/4] Parameter Golf already cloned at $PG_DIR"
fi

if [ ! -f "$PG_TOKENIZER" ]; then
    echo "[1/4] Downloading sp1024 data..."
    cd "$PG_DIR"
    python3 data/fineweb.py --version sp1024
    cd "$REPO_ROOT"
else
    echo "[1/4] sp1024 data already present"
fi

# Optionally limit shards for smoke testing
if [ -n "$TRAIN_SHARDS" ]; then
    echo "[info] Smoke test mode: using first $TRAIN_SHARDS train shards + 1 val shard"
    # Create a temp dir with symlinks to limited shards
    SMOKE_DIR="$REPO_ROOT/.smoke_data"
    rm -rf "$SMOKE_DIR" && mkdir -p "$SMOKE_DIR"
    SHARD_FILES=($(ls "$PG_DATA_DIR"/*.bin | sort))
    # Link first N train shards + last shard (val)
    for i in $(seq 0 $((TRAIN_SHARDS - 1))); do
        ln -s "${SHARD_FILES[$i]}" "$SMOKE_DIR/"
    done
    ln -s "${SHARD_FILES[-1]}" "$SMOKE_DIR/"
    # Copy tokenizer model
    cp "$PG_TOKENIZER" "$SMOKE_DIR/"
    PG_DATA_DIR="$SMOKE_DIR"
    PG_TOKENIZER="$SMOKE_DIR/tok1024.model"
fi

# ---------------------------------------------------------------------------
# Step 2: Run Parameter Golf baseline
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running Parameter Golf baseline (train_gpt.py)..."
echo "------------------------------------------------------------"

PG_LOG="$REPO_ROOT/runs/pgolf_log.txt"

cd "$PG_DIR"
if [ "$NGPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py \
        --input_bin "$PG_DATA_DIR/*.bin" \
        --num_iterations "$NUM_ITERATIONS" \
        --device_batch_size "$DEVICE_BATCH_SIZE" \
        --sequence_length "$MAX_SEQ_LEN" \
        2>&1 | tee "$PG_LOG"
else
    python3 train_gpt.py \
        --input_bin "$PG_DATA_DIR/*.bin" \
        --num_iterations "$NUM_ITERATIONS" \
        --device_batch_size "$DEVICE_BATCH_SIZE" \
        --sequence_length "$MAX_SEQ_LEN" \
        2>&1 | tee "$PG_LOG"
fi
cd "$REPO_ROOT"

# Extract final val_bpb from Parameter Golf log
PG_BPB=$(grep -oP 'val_bpb[:\s=]+\K[0-9.]+' "$PG_LOG" | tail -1 || echo "N/A")
echo ""
echo "Parameter Golf final val_bpb: $PG_BPB"

# ---------------------------------------------------------------------------
# Step 3: Run nanochat on same data
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Running nanochat base_train.py with pretokenized data..."
echo "------------------------------------------------------------"

NC_LOG="$REPO_ROOT/runs/nanochat_log.txt"

cd "$REPO_ROOT/nanochat"
if [ "$NGPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.base_train -- \
        --pretokenized-data-dir "$PG_DATA_DIR" \
        --vocab-size 1024 \
        --tokenizer-path "$PG_TOKENIZER" \
        --depth="$DEPTH" --head-dim=64 --window-pattern=L \
        --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
        --num-iterations="$NUM_ITERATIONS" --eval-every="$EVAL_EVERY" \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        2>&1 | tee "$NC_LOG"
else
    python3 -m scripts.base_train \
        --pretokenized-data-dir "$PG_DATA_DIR" \
        --vocab-size 1024 \
        --tokenizer-path "$PG_TOKENIZER" \
        --depth="$DEPTH" --head-dim=64 --window-pattern=L \
        --max-seq-len="$MAX_SEQ_LEN" --device-batch-size="$DEVICE_BATCH_SIZE" \
        --num-iterations="$NUM_ITERATIONS" --eval-every="$EVAL_EVERY" \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        2>&1 | tee "$NC_LOG"
fi
cd "$REPO_ROOT"

# Extract final val_bpb from nanochat log
NC_BPB=$(grep -oP 'Validation bpb: \K[0-9.]+' "$NC_LOG" | tail -1 || echo "N/A")
echo ""
echo "nanochat final val_bpb: $NC_BPB"

# ---------------------------------------------------------------------------
# Step 4: Print comparison
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " RESULTS — BPB Comparison (lower is better)"
echo "============================================================"
printf "%-20s %s\n" "Engine" "Val BPB"
printf "%-20s %s\n" "---" "---"
printf "%-20s %s\n" "Parameter Golf" "$PG_BPB"
printf "%-20s %s\n" "nanochat (d=$DEPTH)" "$NC_BPB"
echo "============================================================"
echo "Depth: $DEPTH | Iterations: $NUM_ITERATIONS | GPUs: $NGPUS"
echo "Data: FineWeb sp1024 (1024 vocab)"
echo "============================================================"

# Cleanup smoke test dir if it was created
if [ -n "$TRAIN_SHARDS" ] && [ -d "$SMOKE_DIR" ]; then
    rm -rf "$SMOKE_DIR"
fi
