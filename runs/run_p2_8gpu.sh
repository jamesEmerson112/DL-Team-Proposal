#!/bin/bash
# Phase 2 — SmearGate + LQER on 8×H100
# Requires Phase 1 results first to confirm hparam baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEED="${1:-42}"

echo "========================================"
echo "  Phase 2: SmearGate + LQER"
echo "  Seed: $SEED"
echo "  New techniques: SmearGate (BOS-fixed), LQER rank-4 int4"
echo "========================================"
echo ""

bash "$SCRIPT_DIR/run_legal_8gpu.sh" "$SCRIPT_DIR/configs/p2_smear_lqer.env" "$SEED"

echo ""
echo "========================================"
echo "  Phase 2 Complete"
echo "  Compare vs Phase 1 and C6 (1.0805)"
echo "========================================"
