#!/bin/bash
# =============================================================================
# Session 16: Full Pipeline — Phase 1 + 2 + 3 on 2xH100 (6 runs)
#
# Phase 1: EMA deeper sweep (3 runs, ~40 min)
#   R1: EMA=0.995 + WD=0.10 (combo)
#   R2: EMA=0.993
#   R3: EMA=0.990
#
# Phase 2: PreQuantTTT with best EMA from Phase 1 (1 run, ~15 min)
#   R4: PreQuantTTT 21ep (brotli)
#
# Phase 3: Compression improvements (2 runs, ~30 min)
#   R5: PreQuantTTT + lrzip (emb7)
#   R6: PreQuantTTT + lrzip (emb8 — relaxed, recover 0.0017 BPB)
#
# Base: C6 (v2_base + headwise + emb7 + eclip15)
# Usage: bash runs/run_v2_session16_2gpu.sh
# Estimated time: ~1.5 hours total
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
# PHASE 1: EMA Deeper Sweep (3 runs)
# =============================================================================
echo ""
echo "########################################"
echo "  PHASE 1: EMA Deeper Sweep"
echo "########################################"

# R1: EMA=0.995 + WD=0.10 combo
set_c6_base
export EMA_DECAY=0.995
export MUON_WD=0.10
run_one "c6_ema995_wd10" "R1: EMA=0.995 + WD=0.10 combo"

# R2: EMA=0.993
set_c6_base
export EMA_DECAY=0.993
run_one "c6_ema993" "R2: EMA=0.993 (deeper averaging)"

# R3: EMA=0.990
set_c6_base
export EMA_DECAY=0.990
run_one "c6_ema990" "R3: EMA=0.990 (find the floor)"

# --- Phase 1 summary ---
echo ""
echo "============================================"
echo "  PHASE 1 COMPLETE — picking best EMA"
echo "============================================"

BEST_EMA=$(python3 -c "
import os, re
runs = [('0.995', 'c6_ema995_wd10'), ('0.993', 'c6_ema993'), ('0.990', 'c6_ema990')]
best_bpb = float('inf'); best_ema = '0.995'
for ema, run_id in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file): continue
    text = open(log_file).read()
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if not m: m = re.search(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if not m: m = re.search(r'quantized val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m:
        bpb = float(m.group(1))
        if bpb < best_bpb: best_bpb = bpb; best_ema = ema
print(best_ema)
")

# Check if R1 (combo) won — if so, use WD=0.10 too
BEST_WD="0.095"
if [ "$BEST_EMA" = "0.995" ]; then
    BEST_WD="0.10"
fi

echo "  Best EMA from Phase 1: $BEST_EMA (WD=$BEST_WD)"
echo "  Using for Phase 2-3..."

# =============================================================================
# PHASE 2: PreQuantTTT (1 run)
# =============================================================================
echo ""
echo "########################################"
echo "  PHASE 2: PreQuantTTT"
echo "########################################"

set_c6_base
export EMA_DECAY=$BEST_EMA
export MUON_WD=$BEST_WD
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
run_one "c6_prequant_ttt" "R4: PreQuantTTT 21ep (EMA=$BEST_EMA, WD=$BEST_WD)"

# =============================================================================
# PHASE 3: Compression Improvements (2 runs)
# =============================================================================
echo ""
echo "########################################"
echo "  PHASE 3: Compression (lrzip)"
echo "########################################"

# R5: lrzip + emb7
set_c6_base
export EMA_DECAY=$BEST_EMA
export MUON_WD=$BEST_WD
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
export COMPRESSOR=pergroup
run_one "c6_prequant_lrzip_emb7" "R5: PreQuantTTT + lrzip emb7 (EMA=$BEST_EMA)"

# R6: lrzip + emb8 (relaxed compression)
set_c6_base
export EMA_DECAY=$BEST_EMA
export MUON_WD=$BEST_WD
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
export COMPRESSOR=pergroup
export EMBED_BITS=8
export EMBED_CLIP_SIGMAS=20.0
run_one "c6_prequant_lrzip_emb8" "R6: PreQuantTTT + lrzip emb8 (EMA=$BEST_EMA)"

# =============================================================================
# FULL RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  SESSION 16 COMPLETE — ALL 6 RUNS"
echo "============================================"

python3 -c "
import os, re

runs = [
    # Phase 1
    ('R1 ema995+wd10',       'c6_ema995_wd10'),
    ('R2 ema=0.993',         'c6_ema993'),
    ('R3 ema=0.990',         'c6_ema990'),
    # Phase 2
    ('R4 prequant_ttt',      'c6_prequant_ttt'),
    # Phase 3
    ('R5 lrzip+emb7',        'c6_prequant_lrzip_emb7'),
    ('R6 lrzip+emb8',        'c6_prequant_lrzip_emb8'),
]

print()
print(f'{\"Run\":>20} | {\"TTT BPB\":>9} | {\"SW BPB\":>9} | {\"Pre-Q BPB\":>9} | {\"PostPQ BPB\":>9} | {\"Weights\":>12} | Budget?')
print(f'{\"-\"*20} | {\"-\"*9} | {\"-\"*9} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} | -------')

for label, run_id in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'{label:>20} | {\"MISSING\":>9} |           |           |           |              |')
        continue
    text = open(log_file).read()

    ttt_bpb = sw_bpb = preq_bpb = postpq_bpb = None
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = float(m.group(1))
    m = re.search(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: sw_bpb = float(m.group(1))
    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: preq_bpb = float(m.group(1))
    m = re.search(r'post-prequant-ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: postpq_bpb = float(m.group(1))

    quant_bytes = None
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: quant_bytes = int(m.group(1))

    tb = f'{ttt_bpb:.4f}' if ttt_bpb else '?'
    sb = f'{sw_bpb:.4f}' if sw_bpb else '?'
    pb = f'{preq_bpb:.4f}' if preq_bpb else '?'
    ppb = f'{postpq_bpb:.4f}' if postpq_bpb else '?'
    qb = f'{quant_bytes:,}' if quant_bytes else '?'
    budget = 'Yes' if quant_bytes and quant_bytes < 16_000_000 else 'No/?'
    print(f'{label:>20} | {tb:>9} | {sb:>9} | {pb:>9} | {ppb:>9} | {qb:>12} | {budget}')

print()
print(f'  Session 15 baselines:')
print(f'    E1 (EMA=0.995):       1.1562 TTT BPB')
print(f'    C6 (default):         1.1622 TTT BPB')
print(f'  Projected 8xH100:')
print(f'    E1 scaling:           ~1.0745 BPB')
print(f'    + PreQuantTTT (~0.06): ~1.015 BPB')
print(f'    Current SOTA:         1.0136 BPB')
"

echo ""
echo "============================================"
echo "  Next: run best config on 8xH100 3-seed"
echo "============================================"
