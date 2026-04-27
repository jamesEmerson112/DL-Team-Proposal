#!/bin/bash

# ==========================================================================
#  Parameter Golf — Quickstart
#  One command to set up and run the best config on any RunPod.
#
#  Usage (from repo root):
#    bash runs/quickstart.sh
#
#  What it does:
#    1. Detects GPU count
#    2. Installs/upgrades dependencies
#    3. Downloads SP8192 dataset (if needed)
#    4. Runs the best config (sp8192_combo_slim + TTT)
#    5. Prints results vs PG baseline (1.2244 BPB)
#
#  Safe to run multiple times — skips steps already completed.
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# ---- Step 0: Verify we're in the right place ----
if [ ! -f "$PG_DIR/train_gpt.py" ]; then
    echo "ERROR: parameter-golf/train_gpt.py not found."
    echo "  Make sure you cloned with --recurse-submodules:"
    echo "  git clone --recurse-submodules https://github.com/jamesEmerson112/DL-Team-Proposal.git"
    echo "  cd DL-Team-Proposal"
    echo "  bash runs/quickstart.sh"
    exit 1
fi

# ---- Step 1: Detect GPUs ----
if [ -n "${NGPUS:-}" ]; then
    echo "Using NGPUS=$NGPUS (from environment)"
elif command -v nvidia-smi &>/dev/null; then
    NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Auto-detected $NGPUS GPU(s)"
else
    NGPUS=1
    echo "nvidia-smi not found, defaulting to NGPUS=1"
fi
export NGPUS

# ---- Cost/time estimate ----
echo ""
echo "============================================"
echo "  Parameter Golf — Quickstart"
echo "============================================"
echo ""
echo "  GPUs:     $NGPUS"
echo "  Config:   SP8192 combo slim + TTT (best known)"
echo "  Time:     ~12 min training + ~3 min TTT eval"
if [ "$NGPUS" -ge 8 ]; then
    echo "  Est cost: ~\$5 on 8xH100"
    echo "  Expected: ~1.2077 BPB (matches Run 11)"
elif [ "$NGPUS" -ge 2 ]; then
    echo "  Est cost: ~\$1.50 on 2xH100"
    echo "  Expected: ~1.2411 BPB (matches Run A)"
else
    echo "  Est cost: ~\$0.75 on 1xGPU"
    echo "  Expected: higher BPB (fewer steps in 10 min)"
fi
echo "  Baseline: 1.2244 BPB"
echo ""

# ---- Step 2: Install dependencies ----
echo ">> Installing dependencies..."
cd "$PG_DIR"
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install huggingface_hub -q
pip install --upgrade torch -q
echo "   PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "   Dependencies OK"

# ---- Step 3: Download SP8192 dataset ----
SP8192_DATA="$PG_DIR/data/datasets/fineweb10B_sp8192"
if [ -d "$SP8192_DATA" ] && [ "$(ls -1 "$SP8192_DATA"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ">> SP8192 dataset already downloaded, skipping"
else
    echo ">> Downloading SP8192 dataset (~5-10 min)..."
    rm -f "$PG_DIR/data/manifest.json"
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 "$PG_DIR/data/cached_challenge_fineweb.py" --variant sp8192 --train-shards 80
    echo "   Dataset downloaded"
fi

# ---- Step 4: Run best config ----
cd "$REPO_ROOT"
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
export NGPUS=$NGPUS
export RUN_ID="quickstart_${NGPUS}gpu"

echo ""
echo ">> Starting training..."
echo "   Config:  sp8192_combo_slim.env"
echo "   RUN_ID:  $RUN_ID"
echo "   GPUs:    $NGPUS"
echo ""

bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"

echo ""
echo "============================================"
echo "  Quickstart complete!"
echo "  Log: parameter-golf/logs/${RUN_ID}.txt"
echo "============================================"
