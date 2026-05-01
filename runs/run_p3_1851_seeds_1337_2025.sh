#!/bin/bash
set -euo pipefail

# P3: Run seeds 1337 and 2025 sequentially on the same 8×H100 pod
# Run AFTER seed 42 completes via run_p3_1851_8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_DIR="${SCRIPT_DIR}/../parameter-golf"
cd "$PG_DIR"

NGPUS="${NGPUS:-8}"

echo "=== P3: Seeds 1337 + 2025 (sequential, ${NGPUS}×GPU) ==="
echo ""

# --- Seed 1337 ---
echo "=========================================="
echo "  SEED 1337"
echo "=========================================="
set -a
source "${SCRIPT_DIR}/configs/p3_1851_seed1337.env"
set +a

echo "Config: SEED=$SEED CASEOPS_ENABLED=$CASEOPS_ENABLED EMA_DECAY=$EMA_DECAY GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED"
torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

LOGFILE_1337=$(ls -t logs/*.txt 2>/dev/null | head -1)
echo "Seed 1337 log: $LOGFILE_1337"
echo ""

# --- Seed 2025 ---
echo "=========================================="
echo "  SEED 2025"
echo "=========================================="
set -a
source "${SCRIPT_DIR}/configs/p3_1851_seed2025.env"
set +a

echo "Config: SEED=$SEED CASEOPS_ENABLED=$CASEOPS_ENABLED EMA_DECAY=$EMA_DECAY GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED"
torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

LOGFILE_2025=$(ls -t logs/*.txt 2>/dev/null | head -1)
echo "Seed 2025 log: $LOGFILE_2025"
echo ""

# --- Summary ---
echo "=========================================="
echo "  3-SEED SUMMARY"
echo "=========================================="
echo "Seed 1337 log: $LOGFILE_1337"
echo "Seed 2025 log: $LOGFILE_2025"
echo ""
echo "Extract val_bpb from each log + seed 42, then compute mean:"
echo "  python3 -c \"import statistics; bpbs=[SEED42, SEED1337, SEED2025]; print(f'Mean: {statistics.mean(bpbs):.4f} Std: {statistics.stdev(bpbs):.4f}')\""
