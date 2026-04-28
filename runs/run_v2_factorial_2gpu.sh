#!/bin/bash

# V2 Factorial — 3×3: (PR vs RF vs Both) × (None vs Headwise vs Elementwise)
#
# Fork of rank 1's train_gpt.py + our gated attention + ResFormer.
# 9 runs, ~13 min each, ~117 min total on 2×H100.
#
# Usage (from repo root):  bash runs/run_v2_factorial_2gpu.sh

set -uo pipefail

NGPUS="${NGPUS:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/v2_base.env"
TRAIN_SCRIPT="train_gpt_v2.py"

echo "============================================"
echo "  V2 Factorial: 3×3 Sweep — ${NGPUS}×GPU"
echo "  $(date)"
echo "============================================"

# --- Preflight: check FA3 + brotli ---
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found. Install with:"
    echo "  pip install --no-cache-dir https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    exit 1
}
python3 -c "import brotli" 2>/dev/null || {
    echo "ERROR: brotli not found. Install with: pip install brotli"
    exit 1
}
echo "Preflight: FA3 + brotli OK"

cd "$REPO_ROOT"

# --- Helper ---
run_v2() {
    local label="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "############################################################"
    echo ""

    source "$BASE_CONFIG"
    export RUN_ID="$run_id"
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: GATED_ATTN=$GATED_ATTN VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA PARALLEL_RESIDUAL_START=$PARALLEL_RESIDUAL_START"

    cd "$PG_DIR"
    if torchrun --standalone --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" 2>&1 | tee "logs/${run_id}.txt"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
    cd "$REPO_ROOT"
}

# ==========================================
# ROW 1: Parallel Residuals only (rank 1 default)
# ==========================================
echo ""
echo "========== ROW 1: PR ONLY =========="

run_v2 "F1: PR + No Gate (CONTROL)" "v2_pr_none"

run_v2 "F2: PR + Headwise Gate" "v2_pr_head" \
    "GATED_ATTN=headwise"

run_v2 "F3: PR + Elementwise Gate" "v2_pr_elem" \
    "GATED_ATTN=elementwise"

# ==========================================
# ROW 2: ResFormer only (disable PR)
# ==========================================
echo ""
echo "========== ROW 2: RF ONLY =========="

run_v2 "F4: RF + No Gate" "v2_rf_none" \
    "PARALLEL_RESIDUAL_START=999" "VALUE_RESIDUAL_ALPHA=0.5"

run_v2 "F5: RF + Headwise Gate" "v2_rf_head" \
    "PARALLEL_RESIDUAL_START=999" "VALUE_RESIDUAL_ALPHA=0.5" "GATED_ATTN=headwise"

run_v2 "F6: RF + Elementwise Gate" "v2_rf_elem" \
    "PARALLEL_RESIDUAL_START=999" "VALUE_RESIDUAL_ALPHA=0.5" "GATED_ATTN=elementwise"

# ==========================================
# ROW 3: Both PR + RF
# ==========================================
echo ""
echo "========== ROW 3: PR + RF =========="

run_v2 "F7: PR+RF + No Gate" "v2_both_none" \
    "VALUE_RESIDUAL_ALPHA=0.5"

run_v2 "F8: PR+RF + Headwise Gate" "v2_both_head" \
    "VALUE_RESIDUAL_ALPHA=0.5" "GATED_ATTN=headwise"

run_v2 "F9: PR+RF + Elementwise Gate" "v2_both_elem" \
    "VALUE_RESIDUAL_ALPHA=0.5" "GATED_ATTN=elementwise"

# ==========================================
# RESULTS SUMMARY
# ==========================================
echo ""
echo "############################################################"
echo "  V2 FACTORIAL RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('', '', '--- PR ONLY ---'),
    ('F1', 'v2_pr_none',    'PR + No Gate (CTRL)'),
    ('F2', 'v2_pr_head',    'PR + Headwise'),
    ('F3', 'v2_pr_elem',    'PR + Elementwise'),
    ('', '', '--- RF ONLY ---'),
    ('F4', 'v2_rf_none',    'RF + No Gate'),
    ('F5', 'v2_rf_head',    'RF + Headwise'),
    ('F6', 'v2_rf_elem',    'RF + Elementwise'),
    ('', '', '--- PR + RF ---'),
    ('F7', 'v2_both_none',  'PR+RF + No Gate'),
    ('F8', 'v2_both_head',  'PR+RF + Headwise'),
    ('F9', 'v2_both_elem',  'PR+RF + Elementwise'),
]

pg_dir = '$PG_DIR'
na = '---'

print(f\"  {'Run':>3} | {'Config':<22} | {'Pre-Q BPB':>9} | {'Quant BPB':>9} | {'TTT BPB':>8} | {'Size':>10} | {'Steps':>5}\")
print(f\"  {'-'*3} | {'-'*22} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*5}\")

for label, run_id, desc in runs:
    if not run_id:
        print(f'  {desc}')
        continue
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<22} | {na:>9} | {na:>9} | {na:>8} | {na:>10} | {na:>5}')
        continue
    text = open(log_file).read()

    pre_bpb = '?'
    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: pre_bpb = m.group(1)

    quant_bpb = '?'
    m = re.search(r'^quantized val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
    if m: quant_bpb = m.group(1)

    ttt_bpb = '?'
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    sz = '?'
    m = re.search(r'Total submission size.*?: (\d+) bytes', text)
    if m: sz = f'{int(m.group(1))/1e6:.2f}MB'

    steps = '?'
    m = re.search(r'stopping_early.*?step: (\d+)', text)
    if not m:
        vals = re.findall(r'^(\d+)/\d+ val_loss', text, re.MULTILINE)
        if vals: steps = vals[-1]

    print(f'  {label:>3} | {desc:<22} | {pre_bpb:>9} | {quant_bpb:>9} | {ttt_bpb:>8} | {sz:>10} | {steps:>5}')

print()
print('  F1 = rank 1 control (no additions). Best = lowest TTT BPB.')
print('  Our Run 11 (8xH100): 1.2077 | Best 2xH100: E1 1.2338')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
