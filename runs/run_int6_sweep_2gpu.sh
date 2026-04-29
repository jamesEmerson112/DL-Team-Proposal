#!/bin/bash

# Int6 SDClip + Elementwise Gated Attention Dim Sweep — 3 experiments on 2×H100.
#
# Tests int6+brotli quantization (PG ranks 1-9) with elementwise gated attention
# at 3 MODEL_DIM values to find the sweet spot that fits under 16 MB.
#
# E1 (int8+zlib, dim=448) was 1.2338 BPB but 16.67 MB (over by 0.67 MB).
# Int6+brotli should save ~25% on artifact size, making all 3 dims fit.
#
# Compare to: E1 (1.2338 BPB, 16.67 MB) and Run A (1.2411 BPB, 15.03 MB)
#
# Prerequisites:
#   - SP8192 dataset downloaded
#   - pip install -r requirements.txt (includes brotli)
#
# Usage (from repo root):
#   bash runs/run_int6_sweep_2gpu.sh

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# Track results
declare -a RUN_NAMES=()
declare -a RUN_IDS=()
declare -a RUN_STATUS=()

run_experiment() {
    local run_name="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $run_name"
    echo "  RUN_ID: $run_id | GPUs: $NGPUS"
    echo "############################################################"
    echo ""

    RUN_NAMES+=("$run_name")
    RUN_IDS+=("$run_id")

    export NGPUS=$NGPUS
    export RUN_ID="$run_id"
    for override in "$@"; do
        export "$override"
    done

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        RUN_STATUS+=("OK")
    else
        RUN_STATUS+=("FAIL")
        echo "WARNING: $run_name ($run_id) failed, continuing..."
    fi
}

echo "============================================"
echo "  Int6 SDClip + Elementwise Dim Sweep — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# ==========================================================================
# I1: Elementwise dim=448 + int6 (retry E1 with int6 compression)
# E1 was 1.2338 BPB, 16.67 MB (int8). Should fit ~12.5 MB with int6.
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_elementwise_int6_dim448.env"
run_experiment \
    "I1: Elem dim=448 int6" \
    "sp8192_elem_int6_dim448_2gpu"

# ==========================================================================
# I2: Elementwise dim=480 + int6 (bigger model, still under 16 MB?)
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_elementwise_int6_dim480.env"
run_experiment \
    "I2: Elem dim=480 int6" \
    "sp8192_elem_int6_dim480_2gpu"

# ==========================================================================
# I3: Elementwise dim=512 + int6 (full original dim, can int6 make it fit?)
# Run 3 was 17.87 MB with int8. Should be ~13.4 MB with int6.
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_elementwise_int6_dim512.env"
run_experiment \
    "I3: Elem dim=512 int6" \
    "sp8192_elem_int6_dim512_2gpu"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================

echo ""
echo "############################################################"
echo "  INT6 SDCLIP + ELEMENTWISE SWEEP — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

run_ids = [
    ('I1', 'sp8192_elem_int6_dim448_2gpu',  'Elem dim=448 int6',  '1.2338', '16.67'),
    ('I2', 'sp8192_elem_int6_dim480_2gpu',  'Elem dim=480 int6',  '1.2338', '—'),
    ('I3', 'sp8192_elem_int6_dim512_2gpu',  'Elem dim=512 int6',  '1.2338', '—'),
]

pg_dir = '$PG_DIR'

print(f'  {\"Run\":>3} | {\"Config\":<20} | {\"val_bpb\":>8} | {\"E1 ref\":>8} | {\"Delta\":>8} | {\"Steps\":>6} | {\"Size\":>10} | Budget')
print(f'  {\"-\"*3} | {\"-\"*20} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*6} | {\"-\"*10} | ------')

for label, run_id, desc, baseline, old_size in run_ids:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<20} | {\"---\":>8} | {baseline:>8} | {\"---\":>8} | {\"---\":>6} | {\"---\":>10} | ---')
        continue

    text = open(log_file).read()

    bpb = '?'
    # Try int6 TTT first, then int6 roundtrip, then int8 TTT, then int8 roundtrip
    for pat in [
        r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
        r'final_int6_brotli_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
        r'final_int8_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
        r'final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
    ]:
        m = re.search(pat, text)
        if m:
            bpb = m.group(1)
            break
    if bpb == '?':
        vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
        if vals:
            bpb = vals[-1]

    steps = re.findall(r'^step:(\d+)/\d+', text, re.MULTILINE)
    last_step = steps[-1] if steps else '?'

    comp = '?'
    budget = '?'
    # Check for int6+brotli size first, then int8+zlib
    for size_pat in [r'Serialized model int6\+brotli: (\d+) bytes', r'Serialized model int8\+zlib: (\d+) bytes']:
        m = re.search(size_pat, text)
        if m:
            sz = int(m.group(1))
            comp = f'{sz/1e6:.2f}MB'
            budget = 'YES' if sz < 16000000 else 'NO'
            break

    delta = '?'
    try:
        delta = f'{float(bpb) - float(baseline):+.4f}'
    except:
        pass

    print(f'  {label:>3} | {desc:<20} | {bpb:>8} | {baseline:>8} | {delta:>8} | {last_step:>6} | {comp:>10} | {budget}')

print()
print('  Reference: E1 (int8+zlib, elem dim=448) = 1.2338 BPB, 16.67 MB (OVER)')
print('  Reference: Run A (headwise dim=448, TTT) = 1.2411 BPB, 15.03 MB')
print()
print('  PASS = val_bpb < 1.2338 AND size < 16 MB')
"

echo ""
echo "Done. Best run that fits budget + beats E1 (1.2338) → goes to 8×H100."
