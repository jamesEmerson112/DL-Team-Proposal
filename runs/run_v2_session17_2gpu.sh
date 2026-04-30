#!/bin/bash
# =============================================================================
# Session 17: DiffAttn + HybridNorm A/B Tests on 2×H100 (3-4 runs)
#
# N1: C6 + EMA=0.990 + Small Batch (legal combo baseline)
# N2: N1 + Differential Attention (Paper #19, ICLR 2025 Oral)
# N3: N1 + HybridNorm V-norm (Paper #21, NeurIPS 2025)
# N4: N1 + best of N2/N3 (if both help, stack them)
#
# Base: C6 (headwise + emb7 + eclip15) + EMA=0.990 + Small Batch (ga=1)
# Usage: bash runs/run_v2_session17_2gpu.sh
# Estimated time: ~45 min (3 runs × ~13 min + overhead)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

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

    torchrun --standalone --nproc_per_node=2 train_gpt.py
    echo "==> Done. Log: logs/${RUN_ID}.txt"
}

# =============================================================================
# N1: Legal combo baseline — C6 + EMA=0.990 + Small Batch
# =============================================================================
echo ""
echo "########################################"
echo "  N1: Legal combo baseline"
echo "########################################"

source "$REPO_ROOT/runs/configs/v2_n1_combo.env"
export SEED=42
run_one "v2_n1_combo" "N1: C6 + EMA=0.990 + Small Batch (legal combo baseline)"

# =============================================================================
# N2: Differential Attention (Paper #19)
# =============================================================================
echo ""
echo "########################################"
echo "  N2: Differential Attention"
echo "########################################"

source "$REPO_ROOT/runs/configs/v2_n2_diffattn.env"
export SEED=42
run_one "v2_n2_diffattn" "N2: N1 + Differential Attention (two-softmax-subtract)"

# =============================================================================
# N3: HybridNorm V-norm (Paper #21)
# =============================================================================
echo ""
echo "########################################"
echo "  N3: HybridNorm V-norm"
echo "########################################"

source "$REPO_ROOT/runs/configs/v2_n3_hybridnorm.env"
export SEED=42
run_one "v2_n3_hybridnorm" "N3: N1 + HybridNorm V-norm (RMSNorm on V projections)"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo ""
echo "########################################"
echo "  SESSION 17 RESULTS"
echo "########################################"

python3 -c "
import os, re

runs = [
    ('N1', 'v2_n1_combo', 'C6+EMA0.990+SmallBatch'),
    ('N2', 'v2_n2_diffattn', 'N1+DiffAttn'),
    ('N3', 'v2_n3_hybridnorm', 'N1+HybridNorm_V'),
]

print(f\"{'Run':<5} {'Config':<30} {'TTT BPB':<12} {'Steps':<8}\")
print('-'*60)

best_bpb = float('inf')
best_run = ''

for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'{label:<5} {desc:<30} {\"NO LOG\":<12}')
        continue
    text = open(log_file).read()
    # Try TTT BPB first, then sliding window, then plain quantized
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if not m: m = re.search(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if not m: m = re.search(r'quantized val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    bpb = float(m.group(1)) if m else 0
    # Get step count
    steps_m = re.findall(r'step:(\d+)', text)
    steps = steps_m[-1] if steps_m else '?'
    print(f'{label:<5} {desc:<30} {bpb:<12.4f} {steps:<8}')
    if bpb > 0 and bpb < best_bpb:
        best_bpb = bpb
        best_run = label

print()
print(f'Best run: {best_run} ({best_bpb:.4f} BPB)')
print(f'R3 baseline (EMA=0.990 only): 1.1505 BPB')
print(f'B2 baseline (SmallBatch only): 1.1419 BPB')
print(f'C6 baseline (no tuning):       1.1622 BPB')
"

echo ""
echo "==> Session 17 complete!"
