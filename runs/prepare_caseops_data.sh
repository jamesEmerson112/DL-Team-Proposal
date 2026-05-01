#!/bin/bash
set -euo pipefail

# CaseOps Data Prep — run on any GPU pod with parameter-golf repo
# Creates re-tokenized FineWeb shards with lossless capitalization encoding
# Takes ~30-60 min. Output: data/datasets/fineweb10B_sp8192_caseops/
#
# Prerequisites:
#   - parameter-golf repo cloned
#   - sp8192 dataset already downloaded (fineweb_train_*.bin + fineweb_val_*.bin)
#
# Usage:
#   cd /workspace/parameter-golf
#   bash ../runs/prepare_caseops_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_DIR="${SCRIPT_DIR}/../parameter-golf"
cd "$PG_DIR"

echo "=== CaseOps Data Prep ==="
echo "Working dir: $(pwd)"

# --- Step 1: Download CaseOps files from PR #1855 ---
BASE="https://raw.githubusercontent.com/openai/parameter-golf/refs/heads/main"
SOTA="records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611"

if [ ! -f "lossless_caps.py" ]; then
    echo "Downloading lossless_caps.py..."
    curl -fL "$BASE/$SOTA/lossless_caps.py" -o lossless_caps.py
fi

if [ ! -f "prepare_caseops_data.py" ]; then
    echo "Downloading prepare_caseops_data.py..."
    curl -fL "$BASE/$SOTA/prepare_caseops_data.py" -o prepare_caseops_data.py
fi

mkdir -p data/tokenizers
TOKENIZER="data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
if [ ! -f "$TOKENIZER" ]; then
    echo "Downloading CaseOps tokenizer model..."
    curl -fL "$BASE/$SOTA/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
         -o "$TOKENIZER"
fi

# Sanity check (should print Python, not HTML 404)
FIRST_LINE=$(head -1 lossless_caps.py)
if echo "$FIRST_LINE" | grep -qi "html\|404\|not found"; then
    echo "ERROR: Downloaded HTML instead of Python. PR #1855 may not be merged to main."
    echo "Try: gh pr checkout 1855 --repo openai/parameter-golf --detach"
    echo "Then copy files manually from the records directory."
    exit 1
fi
echo "Downloads OK: $(wc -l < lossless_caps.py) lines in lossless_caps.py"

# --- Step 2: Verify sp8192 base data exists ---
if [ ! -f "data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin" ]; then
    echo "ERROR: sp8192 base data not found."
    echo "Run first: MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80"
    exit 1
fi
echo "sp8192 base data: OK"

# --- Step 3: Reprocess shards ---
OUT_DIR="data/datasets/fineweb10B_sp8192_caseops"
if [ -d "$OUT_DIR" ]; then
    echo "Output dir already exists: $OUT_DIR"
    echo "Delete it first if you want to re-run: rm -rf $OUT_DIR"
    exit 0
fi

echo ""
echo "=== Starting CaseOps reprocessing (~30-60 min) ==="
echo ""

python prepare_caseops_data.py \
    --docs data/datasets/fineweb10B_sp8192 \
    --out "$OUT_DIR" \
    --sp "$TOKENIZER"

# --- Step 4: Verify output ---
echo ""
echo "=== CaseOps data created ==="
TRAIN_COUNT=$(ls "$OUT_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$OUT_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_COUNT"
echo "Val shards:   $VAL_COUNT"
echo "Total size:   $(du -sh "$OUT_DIR" | cut -f1)"

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "WARNING: No train shards found. Check prepare_caseops_data.py output for errors."
    exit 1
fi

echo ""
echo "Done. To use CaseOps in training, set:"
echo "  export DATASETS_DIR=$OUT_DIR"
echo "  export TOKENIZER_PATH=$TOKENIZER"
