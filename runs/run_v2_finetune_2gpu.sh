#!/bin/bash
# =============================================================================
# C6 Fine-Tuning Sweep on 2xH100 (19 runs)
#
# Group 1: TTT Epoch x LR grid (9 runs)
# Group 2: QK-Gain sweep (3 runs)
# Group 3: EMA Decay sweep (3 runs)
# Group 4: Weight Decay sweep (3 runs)
# Group 5: Warmdown Frac (1 run)
#
# Base config: C6 = v2_base.env + headwise + emb7 + eclip15
# Usage: bash runs/run_v2_finetune_2gpu.sh
# Estimated time: ~4 hours (19 runs x ~13 min each)
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
# GROUP 1: TTT Epoch x LR Grid (9 runs)
# =============================================================================
# Current: TTT_EPOCHS=3, TTT_LR=0.005

for EPOCHS in 3 5 7; do
    for LR in 0.003 0.005 0.01; do
        set_c6_base
        export TTT_EPOCHS=$EPOCHS
        export TTT_LR=$LR

        # Format LR for ID (remove decimal point)
        LR_TAG=$(echo "$LR" | sed 's/0\.0*//' | sed 's/^/lr/')
        run_one "c6_ttt_e${EPOCHS}_${LR_TAG}" "TTT Grid: epochs=$EPOCHS lr=$LR"
    done
done

# =============================================================================
# GROUP 2: QK-Gain Sweep (3 runs)
# =============================================================================
# Rank 1: QK_GAIN_INIT=5.25

for QKG in 5.5 5.75 6.0; do
    set_c6_base
    export QK_GAIN_INIT=$QKG

    QKG_TAG=$(echo "$QKG" | sed 's/\.//')
    run_one "c6_qkg${QKG_TAG}" "QK-Gain: $QKG (rank 1 = 5.25)"
done

# =============================================================================
# GROUP 3: EMA Decay Sweep (3 runs)
# =============================================================================
# Rank 1: EMA_DECAY=0.9965

for EMA in 0.995 0.997 0.999; do
    set_c6_base
    export EMA_DECAY=$EMA

    EMA_TAG=$(echo "$EMA" | sed 's/0\.//')
    run_one "c6_ema${EMA_TAG}" "EMA Decay: $EMA (rank 1 = 0.9965)"
done

# =============================================================================
# GROUP 4: Weight Decay Sweep (3 runs)
# =============================================================================
# Rank 1: MUON_WD=0.095

for WD in 0.08 0.10 0.11; do
    set_c6_base
    export MUON_WD=$WD

    WD_TAG=$(echo "$WD" | sed 's/0\.//')
    run_one "c6_wd${WD_TAG}" "Muon WD: $WD (rank 1 = 0.095)"
done

# =============================================================================
# GROUP 5: Warmdown Frac (1 run)
# =============================================================================
# Rank 1: WARMDOWN_FRAC=0.72

set_c6_base
export WARMDOWN_FRAC=0.80
run_one "c6_warmdown80" "Warmdown Frac: 0.80 (rank 1 = 0.72)"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  ALL 19 RUNS COMPLETE — SUMMARY"
echo "============================================"

python3 -c "
import os, re

runs = [
    # Group 1: TTT grid
    ('T1 e3 lr0.003',  'c6_ttt_e3_lr003'),
    ('T2 e3 lr0.005',  'c6_ttt_e3_lr005'),
    ('T3 e3 lr0.01',   'c6_ttt_e3_lr01'),
    ('T4 e5 lr0.003',  'c6_ttt_e5_lr003'),
    ('T5 e5 lr0.005',  'c6_ttt_e5_lr005'),
    ('T6 e5 lr0.01',   'c6_ttt_e5_lr01'),
    ('T7 e7 lr0.003',  'c6_ttt_e7_lr003'),
    ('T8 e7 lr0.005',  'c6_ttt_e7_lr005'),
    ('T9 e7 lr0.01',   'c6_ttt_e7_lr01'),
    # Group 2: QK-Gain
    ('Q1 qkg=5.5',     'c6_qkg55'),
    ('Q2 qkg=5.75',    'c6_qkg575'),
    ('Q3 qkg=6.0',     'c6_qkg60'),
    # Group 3: EMA
    ('E1 ema=0.995',   'c6_ema995'),
    ('E2 ema=0.997',   'c6_ema997'),
    ('E3 ema=0.999',   'c6_ema999'),
    # Group 4: WD
    ('W1 wd=0.08',     'c6_wd08'),
    ('W2 wd=0.10',     'c6_wd10'),
    ('W3 wd=0.11',     'c6_wd11'),
    # Group 5: Warmdown
    ('D1 wd_frac=0.80', 'c6_warmdown80'),
]

print()
print(f'  {\"Run\":>20} | {\"TTT BPB\":>9} | {\"SW BPB\":>9} | {\"Pre-Q BPB\":>9} | {\"Weights\":>12} | Budget?')
print(f'  {\"-\"*20} | {\"-\"*9} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} | -------')

for label, run_id in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'  {label:>20} | {\"MISSING\":>9} |           |           |              |')
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
    print(f'  {label:>20} | {tb:>9} | {sb:>9} | {pb:>9} | {qb:>12} | {budget}')

# Find best per group
print()
print(f'  --- Best per group (by TTT BPB) ---')
groups = {
    'TTT Grid': [r for r in runs if r[1].startswith('c6_ttt_')],
    'QK-Gain': [r for r in runs if r[1].startswith('c6_qkg')],
    'EMA': [r for r in runs if r[1].startswith('c6_ema')],
    'WD': [r for r in runs if r[1].startswith('c6_wd')],
    'Warmdown': [r for r in runs if r[1].startswith('c6_warmdown')],
}
for group_name, group_runs in groups.items():
    best_bpb = float('inf')
    best_label = '?'
    for label, run_id in group_runs:
        log_file = f'logs/{run_id}.txt'
        if not os.path.isfile(log_file): continue
        text = open(log_file).read()
        m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
        if m:
            bpb = float(m.group(1))
            if bpb < best_bpb:
                best_bpb = bpb
                best_label = label
    if best_bpb < float('inf'):
        print(f'    {group_name:>12}: {best_label} = {best_bpb:.4f}')
    else:
        print(f'    {group_name:>12}: no results')

print()
print(f'  C6 baseline (2xH100): 1.1622')
print(f'  C6 baseline (8xH100): 1.0805')
"

echo ""
echo "============================================"
echo "  Done. All 19 runs complete."
echo "============================================"
