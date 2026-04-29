#!/bin/bash

# V2 Compression Tuning — Fit F2 (PR + Headwise Gate) under 16 MB budget
#
# F2 weights are 16,007,049 bytes (+7 KB over 16 MB budget).
# Tests compression knobs: embed bits, clip sigmas, combos.
# 8 runs, ~13 min each, ~104 min total on 2xH100.
#
# Usage (from repo root):  bash runs/run_v2_compress_2gpu.sh

set -uo pipefail

NGPUS="${NGPUS:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/v2_base.env"
TRAIN_SCRIPT="train_gpt_v2.py"

echo "============================================"
echo "  V2 Compression Tuning: F2 Budget Fix"
echo "  $(date)"
echo "  GPUs: $NGPUS | Script: $TRAIN_SCRIPT"
echo "============================================"

# --- Preflight ---
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found."
    exit 1
}
python3 -c "import brotli" 2>/dev/null || {
    echo "ERROR: brotli not found."
    exit 1
}
echo "Preflight: FA3 + brotli OK"

cd "$REPO_ROOT"

# --- Helper (same pattern as run_v2_factorial_2gpu.sh) ---
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
    export GATED_ATTN=headwise
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: GATED_ATTN=$GATED_ATTN EMBED_BITS=$EMBED_BITS MATRIX_CLIP_SIGMAS=$MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS=$EMBED_CLIP_SIGMAS"

    cd "$PG_DIR"
    mkdir -p logs
    if torchrun --standalone --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" 2>&1 | tee "logs/${run_id}.txt"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
    cd "$REPO_ROOT"
}

# ==========================================
# Individual knobs
# ==========================================

run_v2 "C1: Embed int7" "v2_f2_emb7" \
    "EMBED_BITS=7"

run_v2 "C2: Embed int6" "v2_f2_emb6" \
    "EMBED_BITS=6"

run_v2 "C3: Matrix clip 10.0" "v2_f2_clip10" \
    "MATRIX_CLIP_SIGMAS=10.0"

run_v2 "C4: Matrix clip 8.0" "v2_f2_clip8" \
    "MATRIX_CLIP_SIGMAS=8.0"

# ==========================================
# Combinations
# ==========================================

run_v2 "C5: Embed7 + clip10" "v2_f2_emb7_clip10" \
    "EMBED_BITS=7" "MATRIX_CLIP_SIGMAS=10.0"

run_v2 "C6: Embed7 + eclip15" "v2_f2_emb7_eclip15" \
    "EMBED_BITS=7" "EMBED_CLIP_SIGMAS=15.0"

run_v2 "C7: Embed7 + clip10 + eclip15" "v2_f2_emb7_clip10_eclip15" \
    "EMBED_BITS=7" "MATRIX_CLIP_SIGMAS=10.0" "EMBED_CLIP_SIGMAS=15.0"

run_v2 "C8: clip10 + eclip15 (keep emb8)" "v2_f2_eclip15_clip10" \
    "MATRIX_CLIP_SIGMAS=10.0" "EMBED_CLIP_SIGMAS=15.0"

# ==========================================
# RESULTS SUMMARY
# ==========================================
echo ""
echo "############################################################"
echo "  COMPRESSION TUNING RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('C1', 'v2_f2_emb7',              'Embed int7'),
    ('C2', 'v2_f2_emb6',              'Embed int6'),
    ('C3', 'v2_f2_clip10',            'Matrix clip 10.0'),
    ('C4', 'v2_f2_clip8',             'Matrix clip 8.0'),
    ('C5', 'v2_f2_emb7_clip10',       'Emb7 + clip10'),
    ('C6', 'v2_f2_emb7_eclip15',      'Emb7 + eclip15'),
    ('C7', 'v2_f2_emb7_clip10_eclip15','Emb7+clip10+eclip15'),
    ('C8', 'v2_f2_eclip15_clip10',    'clip10+eclip15 (emb8)'),
]

pg_dir = '$PG_DIR'
budget = 16_000_000
na = '---'

print(f\"  {'Run':>3} | {'Config':<22} | {'TTT BPB':>9} | {'Weights':>12} | {'vs Budget':>10} | {'vs F2':>8} | {'vs F1':>8}\")
print(f\"  {'-'*3} | {'-'*22} | {'-'*9} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*8}\")

f2_bpb = 1.16361365
f1_bpb = 1.16411794

for label, run_id, desc in runs:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<22} | {na:>9} | {na:>12} | {na:>10} | {na:>8} | {na:>8}')
        continue
    text = open(log_file).read()

    ttt_bpb = '?'
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    wt = '?'
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: wt = int(m.group(1))

    if ttt_bpb != '?' and wt != '?':
        wt_mb = f'{wt/1e6:.3f}MB'
        delta_budget = f'{(wt - budget)/1e3:+.1f}KB'
        delta_f2 = f'{float(ttt_bpb) - f2_bpb:+.5f}'
        delta_f1 = f'{float(ttt_bpb) - f1_bpb:+.5f}'
        ok = 'PASS' if wt < budget and float(ttt_bpb) < f1_bpb else 'FAIL'
        print(f'  {label:>3} | {desc:<22} | {ttt_bpb:>9} | {wt_mb:>12} | {delta_budget:>10} | {delta_f2:>8} | {delta_f1:>8} | {ok}')
    else:
        print(f'  {label:>3} | {desc:<22} | {ttt_bpb:>9} | {str(wt):>12} | {na:>10} | {na:>8} | {na:>8}')

print()
print('  Baselines:')
print(f'    F2 (PR+headwise, original): TTT 1.16361, weights 16,007,049 bytes (+7KB over)')
print(f'    F1 (PR, no gate, control):  TTT 1.16412, weights 15,985,988 bytes (under)')
print(f'    Budget: 16,000,000 bytes')
print(f'    PASS = under budget AND beats F1')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
