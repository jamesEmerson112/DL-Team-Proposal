#!/bin/bash
# =============================================================================
# Session 16 Phase 2-3: Eval Techniques + Compression (4 runs)
#
# Run 4: Best EMA + PreQuantTTT (21 epochs AdamW on val before GPTQ)
# Run 5: Best EMA + PreQuantTTT + sliding-window eval (already enabled)
#        (Note: sliding-window is already on by default, TTT uses it too.
#         This run just confirms the full stack works.)
# Run 6a: Best config + per-group lrzip (emb7)
# Run 6b: Best config + per-group lrzip (emb8 — relaxed compression)
#
# Usage: bash runs/run_v2_eval_techniques_2gpu.sh [EMA_DECAY] [MUON_WD]
#   defaults: EMA_DECAY=0.995 MUON_WD=0.095
#   After Phase 1, substitute the best EMA/WD values.
# Estimated time: ~1 hour (4 runs x ~15 min, longer due to PreQuantTTT)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# Allow overriding EMA/WD from command line
BEST_EMA="${1:-0.995}"
BEST_WD="${2:-0.095}"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
echo "==> Using EMA_DECAY=$BEST_EMA, MUON_WD=$BEST_WD"
mkdir -p logs

# Helper: set C6 base config with best EMA/WD
set_best_base() {
    source "$REPO_ROOT/runs/configs/v2_base.env"
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    export SEED=42
    export EMA_DECAY=$BEST_EMA
    export MUON_WD=$BEST_WD
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
# RUN 4: PreQuantTTT (21 epochs AdamW on val set before GPTQ)
# =============================================================================
set_best_base
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
run_one "c6_prequant_ttt" "PreQuantTTT 21ep (EMA=$BEST_EMA, WD=$BEST_WD)"

# =============================================================================
# RUN 6a: Per-group lrzip compression (emb7)
# =============================================================================
set_best_base
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
export COMPRESSOR=pergroup
run_one "c6_prequant_lrzip_emb7" "PreQuantTTT + lrzip (emb7, EMA=$BEST_EMA)"

# =============================================================================
# RUN 6b: Per-group lrzip + relaxed embedding (emb8)
# =============================================================================
set_best_base
export PREQUANT_TTT_ENABLED=1
export PREQUANT_TTT_EPOCHS=21
export PREQUANT_TTT_LR=5e-4
export PREQUANT_TTT_LR_END=5e-5
export COMPRESSOR=pergroup
export EMBED_BITS=8
export EMBED_CLIP_SIGMAS=20.0
run_one "c6_prequant_lrzip_emb8" "PreQuantTTT + lrzip + emb8 (relaxed compression, EMA=$BEST_EMA)"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  ALL RUNS COMPLETE — SUMMARY"
echo "============================================"

python3 -c "
import os, re

runs = [
    ('R4 prequant_ttt',      'c6_prequant_ttt'),
    ('R6a lrzip+emb7',       'c6_prequant_lrzip_emb7'),
    ('R6b lrzip+emb8',       'c6_prequant_lrzip_emb8'),
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
print(f'  Expected improvements:')
print(f'    PreQuantTTT: ~0.06 BPB drop from pre-Q to post-prequant-ttt')
print(f'    lrzip: ~236 KB smaller than brotli')
print(f'    emb8: +0.0017 BPB recovery from relaxed compression')
"

echo ""
echo "============================================"
echo "  Done."
echo "============================================"
