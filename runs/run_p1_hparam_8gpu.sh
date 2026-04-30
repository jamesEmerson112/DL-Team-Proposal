#!/bin/bash
# Phase 1 — SOTA Hparam Overrides on 8×H100
# Runs P1a (NUM_LOOPS=2) and P1b (NUM_LOOPS=3) sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEED="${1:-42}"

echo "========================================"
echo "  Phase 1: SOTA Hparam Overrides"
echo "  Seed: $SEED"
echo "  P1a: SOTA hparams, NUM_LOOPS=2 (3 passes)"
echo "  P1b: SOTA hparams, NUM_LOOPS=3 (4 passes)"
echo "========================================"
echo ""

# P1a — SOTA hparams, default loop count
echo ">>> Starting P1a (NUM_LOOPS=2)..."
bash "$SCRIPT_DIR/run_legal_8gpu.sh" "$SCRIPT_DIR/configs/p1a_hparam_sota.env" "$SEED"
echo ""

# P1b — SOTA hparams + increased loop count
echo ">>> Starting P1b (NUM_LOOPS=3)..."
bash "$SCRIPT_DIR/run_legal_8gpu.sh" "$SCRIPT_DIR/configs/p1b_hparam_sota_loop3.env" "$SEED"
echo ""

echo "========================================"
echo "  Phase 1 Complete"
echo "  Compare P1a vs P1b vs C6 (1.0805)"
echo "========================================"
