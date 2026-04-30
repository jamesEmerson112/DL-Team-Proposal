#!/bin/bash
# Run a single legal experiment on 8×H100
# Usage: bash runs/run_legal_8gpu.sh runs/configs/legal_ema090.env [SEED]
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.env> [seed]}"
SEED="${2:-42}"

# Find parameter-golf directory
PG_DIR="$(cd "$(dirname "$0")/.." && pwd)/parameter-golf"
if [ ! -f "$PG_DIR/train_gpt.py" ]; then
    PG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/parameter-golf"
fi
if [ ! -f "$PG_DIR/train_gpt.py" ]; then
    echo "ERROR: Cannot find parameter-golf/train_gpt.py"
    exit 1
fi

# Source config
source "$CONFIG"
export SEED="$SEED"

echo "============================================"
echo "  Legal 8×H100 Run"
echo "  Config: $CONFIG"
echo "  RUN_ID: $RUN_ID"
echo "  SEED: $SEED"
echo "  EMA_DECAY: $EMA_DECAY"
echo "  GATED_ATTN: $GATED_ATTN"
echo "  PREQUANT_TTT_ENABLED: $PREQUANT_TTT_ENABLED"
echo "  GRAD_ACCUM_STEPS: ${GRAD_ACCUM_STEPS:-default}"
echo "  TRAIN_BATCH_TOKENS: ${TRAIN_BATCH_TOKENS:-default}"
echo "  MLP_CLIP_SIGMAS: ${MLP_CLIP_SIGMAS:-$MATRIX_CLIP_SIGMAS}"
echo "  EMBED_CLIP_SIGMAS: $EMBED_CLIP_SIGMAS"
echo "============================================"

NGPUS="${NGPUS:-8}"
LOG_DIR="$PG_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}_seed${SEED}.txt"

cd "$PG_DIR"

torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "  Run complete: $RUN_ID (seed $SEED)"
echo "  Log: $LOG_FILE"
echo "============================================"

# Extract key metrics from log
echo ""
echo "=== RESULTS ==="
grep -E "^(stopping_early|quantized_sliding_window|quantized_ttt|Total submission size|pre-quantization post-ema)" "$LOG_FILE" || true
echo "================"
