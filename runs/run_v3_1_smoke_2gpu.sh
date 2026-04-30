#!/bin/bash
#
# V3.1 Smoke Test — 2×H100, single run, ~13 min.
#
# Purpose: verify train_gpt_v3_1.py trains, quantizes, and evaluates cleanly
# end-to-end with the new hparams and per-group GPTQ clip sigmas.
#
# This is a structural smoke test, not a BPB-comparison run. Expected
# final TTT BPB on 2×H100 is ~1.163 (similar to F1 = 1.1641). If you
# see NaNs, crashes, or BPB > 1.30, investigate before the 8×H100 run.
#
# Usage (from repo root):
#   bash runs/run_v3_1_smoke_2gpu.sh
#
# Env overrides:
#   NGPUS=N                    # default 2
#   GATED_ATTN=headwise        # match C6/F2 config (optional)

set -uo pipefail

NGPUS="${NGPUS:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/v3_1_base.env"
TRAIN_SCRIPT="train_gpt_v3_1.py"
RUN_ID="${RUN_ID:-v3_1_smoke}"

echo "============================================"
echo "  V3.1 Smoke Test — ${NGPUS}×GPU"
echo "  $(date)"
echo "============================================"

# --- Preflight: FA3 + brotli ---
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found. Install with:"
    echo "  pip install --no-cache-dir https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    exit 1
}
python3 -c "import brotli" 2>/dev/null || {
    echo "ERROR: brotli not found. Install with: pip install brotli"
    exit 1
}

# --- Preflight: data shards present ---
if ! ls "$PG_DIR"/data/datasets/fineweb10B_sp8192/fineweb_train_*.bin >/dev/null 2>&1; then
    echo "ERROR: SP8192 train shards not found at $PG_DIR/data/datasets/fineweb10B_sp8192/"
    echo "Run from parameter-golf/:"
    echo "  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80"
    exit 1
fi

# --- Preflight: train_gpt_v3_1.py present ---
if [ ! -f "$PG_DIR/$TRAIN_SCRIPT" ]; then
    echo "ERROR: $PG_DIR/$TRAIN_SCRIPT not found"
    exit 1
fi

echo "Preflight: FA3 + brotli + shards + script OK"

# --- Load v3.1 base config ---
source "$BASE_CONFIG"
export RUN_ID

# Allow command-line override of GATED_ATTN (default: none)
GATED_ATTN="${GATED_ATTN:-$GATED_ATTN}"
export GATED_ATTN

echo ""
echo "--- Config summary ---"
echo "  RUN_ID=$RUN_ID"
echo "  GATED_ATTN=$GATED_ATTN"
echo "  VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA"
echo "  WARMDOWN_FRAC=$WARMDOWN_FRAC (v3.1: 0.72 -> 0.85)"
echo "  MIN_LR=$MIN_LR                (v3.1: 0.0 -> 0.10)"
echo "  BETA2=$BETA2                  (v3.1: 0.95 -> 0.99)"
echo "  MLP_CLIP_SIGMAS=$MLP_CLIP_SIGMAS       (v3.1: new knob)"
echo "  ATTN_CLIP_SIGMAS=$ATTN_CLIP_SIGMAS     (v3.1: new knob)"
echo "  EMBED_CLIP_SIGMAS=$EMBED_CLIP_SIGMAS   (v3.1: 20.0 -> 14.0)"
echo "  CASEOPS_ENABLED=$CASEOPS_ENABLED"
echo "  MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo ""

# --- Run ---
mkdir -p "$PG_DIR/logs"
cd "$PG_DIR"

LOG_FILE="logs/${RUN_ID}.txt"
echo "Streaming to $PG_DIR/$LOG_FILE"
echo ""

if torchrun --standalone --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "##### V3.1 Smoke: OK #####"
else
    echo ""
    echo "##### V3.1 Smoke: FAILED (see $PG_DIR/$LOG_FILE) #####"
    exit 1
fi

cd "$REPO_ROOT"

# --- Summary ---
echo ""
echo "--- Results ---"
python3 - <<PY
import os, re

log_file = "$PG_DIR/$LOG_FILE"
text = open(log_file).read() if os.path.isfile(log_file) else ""

def grep(pat, flags=0):
    m = re.search(pat, text, flags)
    return m.group(1) if m else "?"

pre_bpb   = grep(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)')
quant_bpb = grep(r'^quantized val_loss:[\d.]+ val_bpb:([\d.]+)', re.MULTILINE)
slide_bpb = grep(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)')
ttt_bpb   = grep(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)')
size      = grep(r'Total submission size.*?: (\d+) bytes')
steps     = grep(r'stopping_early.*?step: (\d+)')

print(f"  pre-quant    : {pre_bpb}")
print(f"  quantized    : {quant_bpb}")
print(f"  sliding      : {slide_bpb}")
print(f"  TTT          : {ttt_bpb}")
print(f"  artifact     : {size} bytes" + (f" ({int(size)/1e6:.2f} MB)" if size.isdigit() else ""))
print(f"  steps        : {steps}")
print()
print("  Sanity checks:")
if ttt_bpb != "?" and float(ttt_bpb) < 1.30:
    print(f"    TTT BPB {ttt_bpb} in expected range (target < 1.17 on 2xH100)")
else:
    print(f"    WARNING: TTT BPB {ttt_bpb} is unexpectedly high or missing")
if size.isdigit() and int(size) < 16_000_000:
    print(f"    artifact fits under 16 MB")
else:
    print(f"    WARNING: artifact {size} bytes exceeds 16 MB budget")
PY

echo ""
echo "Done. Full log: $PG_DIR/$LOG_FILE"
