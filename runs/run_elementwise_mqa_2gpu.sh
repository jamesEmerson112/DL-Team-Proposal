#!/bin/bash

# Elementwise Gated Attention + MQA — 4 experiments on 2×H100.
#
# Tests two techniques that busted the 16 MB budget at dim=512:
#   - Elementwise gated attention (best per-step BPB but +1.87 MB over)
#   - MQA (faster per step but +0.84 MB over)
# Both tested at reduced MODEL_DIM to find the sweet spot.
#
# Compare to: Run A (headwise, GQA, dim=448, TTT) = 1.2411 BPB, 15.03 MB
#
# Prerequisites:
#   - SP8192 dataset downloaded
#   - pip install -r requirements.txt, torch upgraded
#
# Usage (from repo root):
#   bash runs/run_elementwise_mqa_2gpu.sh

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
echo "  Elementwise + MQA Sweep — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# ==========================================================================
# E1: Elementwise gated attention, dim=448 (GQA)
# BPB ceiling — does elementwise fit at the slim dim?
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_elementwise_dim448.env"
run_experiment \
    "E1: Elementwise dim=448 (GQA)" \
    "sp8192_elem_dim448_2gpu"

# ==========================================================================
# E2: Elementwise gated attention, dim=416 (GQA)
# Conservative dim — should fit under 16 MB
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_elementwise_dim416.env"
run_experiment \
    "E2: Elementwise dim=416 (GQA)" \
    "sp8192_elem_dim416_2gpu"

# ==========================================================================
# E3: MQA (1 KV head), dim=448, headwise gated attention
# MQA on SP8192 combo slim — never tested on SP8192 before
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_mqa_dim448.env"
run_experiment \
    "E3: MQA dim=448 (headwise)" \
    "sp8192_mqa_dim448_2gpu"

# ==========================================================================
# E4: MQA + Elementwise, dim=416
# Combo: MQA frees KV params, elementwise adds gate quality
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_mqa_elementwise_dim416.env"
run_experiment \
    "E4: MQA + Elementwise dim=416" \
    "sp8192_mqa_elem_dim416_2gpu"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================

echo ""
echo "############################################################"
echo "  ELEMENTWISE + MQA SWEEP — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

run_ids = [
    ('E1', 'sp8192_elem_dim448_2gpu',     'Elem dim=448 GQA',      '1.2411'),
    ('E2', 'sp8192_elem_dim416_2gpu',     'Elem dim=416 GQA',      '1.2411'),
    ('E3', 'sp8192_mqa_dim448_2gpu',      'MQA dim=448 headwise',  '1.2411'),
    ('E4', 'sp8192_mqa_elem_dim416_2gpu', 'MQA+Elem dim=416',      '1.2411'),
]

pg_dir = '$PG_DIR'

print(f'  {\"Run\":>3} | {\"Config\":<24} | {\"val_bpb\":>8} | {\"Baseline\":>8} | {\"Delta\":>8} | {\"Steps\":>6} | {\"Size\":>10} | Budget')
print(f'  {\"-\"*3} | {\"-\"*24} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*6} | {\"-\"*10} | ------')

for label, run_id, desc, baseline in run_ids:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<24} | {\"---\":>8} | {baseline:>8} | {\"---\":>8} | {\"---\":>6} | {\"---\":>10} | ---')
        continue

    text = open(log_file).read()

    bpb = '?'
    m = re.search(r'final_int8_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m:
        bpb = m.group(1)
    else:
        m = re.search(r'final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
        if m:
            bpb = m.group(1)
        else:
            vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
            if vals:
                bpb = vals[-1]

    steps = re.findall(r'^step:(\d+)/\d+', text, re.MULTILINE)
    last_step = steps[-1] if steps else '?'

    comp = '?'
    budget = '?'
    m = re.search(r'Serialized model int8\+zlib: (\d+) bytes', text)
    if m:
        sz = int(m.group(1))
        comp = f'{sz/1e6:.2f}MB'
        budget = 'YES' if sz < 16000000 else 'NO'

    delta = '?'
    try:
        delta = f'{float(bpb) - float(baseline):+.4f}'
    except:
        pass

    print(f'  {label:>3} | {desc:<24} | {bpb:>8} | {baseline:>8} | {delta:>8} | {last_step:>6} | {comp:>10} | {budget}')

print()
print('  Baseline: Run A (headwise GQA, dim=448, TTT) = 1.2411 BPB, 15.03 MB')
print()
print('  PASS = val_bpb < 1.2411 AND size < 16 MB')
"

echo ""
echo "Done. Best run that fits budget + beats 1.2411 → goes to 8×H100."
