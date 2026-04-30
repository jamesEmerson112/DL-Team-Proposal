#!/bin/bash
# =============================================================================
# Paper #22 (Peri-LN) + Paper #15 (Small Batch Size) — 2xH100
#
# 3 runs (no baseline — use Session 15 R0 = 1.1572 BPB):
#   R1: Peri-LN (output norms on attn + MLP)
#   R2: Small Batch ga=1 (no grad accumulation)
#   R3: Small Batch ga=1 + beta2=0.99 (paper's recommended scaling)
#
# Usage (from parameter-golf/ directory):
#   bash ../runs/run_v2_paper22_paper15_2gpu.sh
#
# Estimated time: ~39 min (3 runs x ~13 min each)
# =============================================================================

set -uo pipefail

NGPUS="${NGPUS:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/v2_base.env"
TRAIN_SCRIPT="train_gpt_v2.py"

echo "============================================"
echo "  Paper #22 + #15: Peri-LN + Small Batch"
echo "  $(date)"
echo "  GPUs: $NGPUS"
echo "============================================"

cd "$PG_DIR"
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || { echo "ERROR: FA3 not found."; exit 1; }
python3 -c "import brotli" 2>/dev/null || { echo "ERROR: brotli not found."; exit 1; }
echo "Preflight: FA3 + brotli OK"
mkdir -p logs

run_v2() {
    local label="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "  $(date)"
    echo "############################################################"
    echo ""

    source "$BASE_CONFIG"
    export RUN_ID="$run_id"
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: GATED_ATTN=$GATED_ATTN PERI_LN=$PERI_LN GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-default} BETA2=${BETA2:-0.95} MUON_BETA2=${MUON_BETA2:-0.95} RUN_ID=$RUN_ID"

    if torchrun --standalone --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" 2>&1 | tee "logs/${run_id}.txt"; then
        echo "==> $label: OK"
    else
        echo "==> $label: FAILED (check logs/$run_id.txt)"
    fi
}

# R1: Peri-LN
run_v2 "R1: Peri-LN" "v2_p22p15_r1_peri_ln" \
    "PERI_LN=1"

# R2: Small Batch ga=1
run_v2 "R2: Small Batch ga=1" "v2_p22p15_r2_ga1" \
    "GRAD_ACCUM_STEPS=1"

# R3: Small Batch ga=1 + beta2=0.99
run_v2 "R3: Small Batch ga=1 + b2=0.99" "v2_p22p15_r3_ga1_b299" \
    "GRAD_ACCUM_STEPS=1" "BETA2=0.99" "MUON_BETA2=0.99"

# RESULTS
echo ""
echo "############################################################"
echo "  ALL 3 RUNS COMPLETE — RESULTS SUMMARY"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('R1', 'v2_p22p15_r1_peri_ln',   'Peri-LN'),
    ('R2', 'v2_p22p15_r2_ga1',       'Small Batch ga=1'),
    ('R3', 'v2_p22p15_r3_ga1_b299',  'Small Batch ga=1+b2=.99'),
]

print(f\"  {'Run':>3} | {'Config':<25} | {'Params':>8} | {'Pre-Q BPB':>9} | {'TTT BPB':>8} | {'Size':>10} | {'Steps':>5}\")
print(f\"  {'-'*3} | {'-'*25} | {'-'*8} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*5}\")

for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<25} | {\"---\":>8} | {\"---\":>9} | {\"---\":>8} | {\"---\":>10} | {\"---\":>5}')
        continue
    text = open(log_file).read()

    params = '?'
    m = re.search(r'model_params:(\d+)', text)
    if m: params = f'{int(m.group(1))/1e6:.1f}M'

    pre_bpb = '?'
    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: pre_bpb = m.group(1)

    ttt_bpb = '?'
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    sz = '?'
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: sz = f'{int(m.group(1))/1e6:.2f}MB'

    steps = '?'
    m = re.search(r'stopping_early.*?step: (\d+)', text)
    if not m:
        vals = re.findall(r'(\d+)/\d+ val_loss:', text)
        if vals: steps = vals[-1]

    print(f'  {label:>3} | {desc:<25} | {params:>8} | {pre_bpb:>9} | {ttt_bpb:>8} | {sz:>10} | {steps:>5}')

baseline = 1.1572
print()
print(f'  Session 15 R0 baseline: {baseline:.4f} TTT BPB')
print()

for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file): continue
    text = open(log_file).read()
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m:
        bpb = float(m.group(1))
        delta = bpb - baseline
        verdict = 'BETTER' if delta < -0.001 else 'WORSE' if delta > 0.001 else 'NEUTRAL'
        print(f'  {label} ({desc}): {bpb:.4f} ({delta:+.4f}) — {verdict}')
"

echo ""
echo "============================================"
echo "  Done. Logs in parameter-golf/logs/"
echo "============================================"
