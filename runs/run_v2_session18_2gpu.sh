#!/bin/bash
# =============================================================================
# Session 18: Cross-Seq Attn + Schedule-Free on 2×H100 (2 runs)
#
# S1: N1 base + Cross-Sequence Attention (eval-only, proven on PG leaderboard)
# S2: N1 base + Schedule-Free Optimizer (Paper #23, replaces AdamW)
#
# Base: N1 = C6 + EMA=0.990 + Small Batch (1.1368 BPB, Session 17 best)
# Usage: bash runs/run_v2_session18_2gpu.sh
# Estimated time: ~26 min (2 runs × ~13 min)
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
# S1: Cross-Sequence Attention (eval-only)
# =============================================================================
echo ""
echo "########################################"
echo "  S1: Cross-Sequence Attention"
echo "########################################"

source "$REPO_ROOT/runs/configs/v2_s1_crossseq.env"
export SEED=42
run_one "v2_s1_crossseq" "S1: N1 + Cross-Sequence Attention (eval-only KV cache)"

# =============================================================================
# S2: Schedule-Free Optimizer (Paper #23)
# =============================================================================
echo ""
echo "########################################"
echo "  S2: Schedule-Free Optimizer"
echo "########################################"

source "$REPO_ROOT/runs/configs/v2_s2_schedulefree.env"
export SEED=42
run_one "v2_s2_schedulefree" "S2: N1 + Schedule-Free AdamW (tok+scalar only)"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo ""
echo "########################################"
echo "  SESSION 18 RESULTS"
echo "########################################"

python3 -c "
import os, re

runs = [
    ('S1', 'v2_s1_crossseq', 'N1+CrossSeqAttn'),
    ('S2', 'v2_s2_schedulefree', 'N1+ScheduleFree'),
]

print(f\"{'Run':<5} {'Config':<25} {'TTT BPB':<12} {'CrossSeq BPB':<14} {'Steps':<8}\")
print('-'*70)

for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'{label:<5} {desc:<25} {\"NO LOG\":<12}')
        continue
    text = open(log_file).read()
    # TTT BPB
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    ttt_bpb = float(m.group(1)) if m else 0
    # Cross-seq BPB
    m2 = re.search(r'quantized_crossseq val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    cs_bpb = f'{float(m2.group(1)):.4f}' if m2 else 'N/A'
    # Steps
    steps_m = re.findall(r'step:(\d+)', text)
    steps = steps_m[-1] if steps_m else '?'
    print(f'{label:<5} {desc:<25} {ttt_bpb:<12.4f} {cs_bpb:<14} {steps:<8}')

print()
print(f'N1 baseline (Session 17):  1.1368 BPB')
print(f'C6 baseline (no tuning):   1.1622 BPB')
"

echo ""
echo "==> Session 18 complete!"
