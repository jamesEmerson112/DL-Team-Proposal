#!/bin/bash
set -euo pipefail

# P3: PR #1851 fork + our 3 novel contributions (8×H100)
# Novel: headwise gated attention + EMA=0.990 + small batch
# Base: PR #1851 (1.0613 BPB, brotli, CaseOps+phased TTT)
# Expected: ~1.034 BPB if deltas hold

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_DIR="${SCRIPT_DIR}/../parameter-golf"
cd "$PG_DIR"

echo "=== P3: PR #1851 + Headwise Gate + EMA=0.990 + Small Batch ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo ""

# Source config (use set -a to export all vars for torchrun subprocesses)
set -a
source "${SCRIPT_DIR}/configs/p3_1851_headwise_ema_smallbatch.env"
set +a

echo "Config:"
echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED"
echo "  SPARSE_ATTN_GATE_ENABLED=$SPARSE_ATTN_GATE_ENABLED"
echo "  EMA_DECAY=$EMA_DECAY"
echo "  GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS"
echo "  TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS"
echo ""

NGPUS="${NGPUS:-8}"
torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

echo ""
echo "=== P3 Complete ==="
echo "Compare: PR #1851 = 1.0613, SOTA = 1.0136"

# Summary extraction
LOGFILE=$(ls -t logs/*.txt 2>/dev/null | head -1)
if [ -n "$LOGFILE" ]; then
    echo ""
    echo "=== Results from $LOGFILE ==="
    grep -E "val_loss|val_bpb|model_params|stopping|ema:|gated_attn|submission" "$LOGFILE" | tail -20
fi
