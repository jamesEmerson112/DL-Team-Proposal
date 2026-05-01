#!/bin/bash
# =============================================================================
# Session 16 Phase 1: EMA Deeper Sweep + Combo (3 runs)
#
# Run 1: EMA=0.995 + WD=0.10 (stack the two Session 15 winners)
# Run 2: EMA=0.993 (deeper averaging)
# Run 3: EMA=0.990 (find the floor)
#
# Base config: C6 = v2_base.env + headwise + emb7 + eclip15
# Usage: bash runs/run_v2_finetune2_2gpu.sh
# Estimated time: ~40 min (3 runs x ~13 min each)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

# Helper: set C6 base config
set_c6_base() {
    source "$REPO_ROOT/runs/configs/v2_base.env"
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    export SEED=42
}

# Helper: run one experiment
run_one() {
    local run_id="$1"
    local desc="$2"
    export RUN_ID="$run_id"

    echo ""
    echo "============================================"
    echo "  $desc"
    echo "  RUN_ID=$RUN_ID"
    echo "============================================"
    echo ""

    torchrun --standalone --nproc_per_node=2 train_gpt_v2.py
    echo "==> Done. Log: logs/${RUN_ID}.txt"
}

# =============================================================================
# RUN 1: EMA=0.995 + WD=0.10 combo
# =============================================================================
set_c6_base
export EMA_DECAY=0.995
export MUON_WD=0.10
run_one "c6_ema995_wd10" "EMA=0.995 + WD=0.10 combo (Session 15 winners stacked)"

# =============================================================================
# RUN 2: EMA=0.993
# =============================================================================
set_c6_base
export EMA_DECAY=0.993
run_one "c6_ema993" "EMA=0.993 (deeper averaging, rank 1 = 0.9965)"

# =============================================================================
# RUN 3: EMA=0.990
# =============================================================================
set_c6_base
export EMA_DECAY=0.990
run_one "c6_ema990" "EMA=0.990 (find the floor, rank 1 = 0.9965)"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  ALL 3 RUNS COMPLETE — SUMMARY"
echo "============================================"

python3 -c "
import os, re

runs = [
    ('R1 ema995+wd10', 'c6_ema995_wd10'),
    ('R2 ema=0.993',   'c6_ema993'),
    ('R3 ema=0.990',   'c6_ema990'),
]

print()
print(f'{\"Run\":>20} | {\"TTT BPB\":>9} | {\"SW BPB\":>9} | {\"Pre-Q BPB\":>9} | {\"Weights\":>12} | Budget?')
print(f'{\"-\"*20} | {\"-\"*9} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} | -------')

for label, run_id in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'{label:>20} | {\"MISSING\":>9} |           |           |              |')
        continue
    text = open(log_file).read()

    ttt_bpb = sw_bpb = preq_bpb = None
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = float(m.group(1))
    m = re.search(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: sw_bpb = float(m.group(1))
    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: preq_bpb = float(m.group(1))

    quant_bytes = None
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: quant_bytes = int(m.group(1))

    tb = f'{ttt_bpb:.4f}' if ttt_bpb else '?'
    sb = f'{sw_bpb:.4f}' if sw_bpb else '?'
    pb = f'{preq_bpb:.4f}' if preq_bpb else '?'
    qb = f'{quant_bytes:,}' if quant_bytes else '?'
    budget = 'Yes' if quant_bytes and quant_bytes < 16_000_000 else 'No/?'
    print(f'{label:>20} | {tb:>9} | {sb:>9} | {pb:>9} | {qb:>12} | {budget}')

print()
print(f'  Session 15 baselines:')
print(f'    E1 (EMA=0.995):       1.1562')
print(f'    W2 (WD=0.10):         1.1619')
print(f'    C6 (default):         1.1622')
"

echo ""
echo "============================================"
echo "  Done. All 3 runs complete."
echo "============================================"
