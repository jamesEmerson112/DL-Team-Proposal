#!/bin/bash
# =============================================================================
# Full Stack: Small Batch + EMA=0.990 + PreQuantTTT (1 run, ~15 min)
#
# Combines the 3 best techniques:
#   B2: Small Batch (ga=1, batch÷4) — -0.015 BPB
#   R3: EMA=0.990 — -0.0117 BPB
#   R4: PreQuantTTT 21ep — -0.1435 BPB
#
# Base: C6 (v2_base + headwise + emb7 + eclip15)
# Usage: bash runs/run_v2_fullstack_2gpu.sh
# Override GPUs: NGPUS=8 bash runs/run_v2_fullstack_2gpu.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

# --- Load configs in order: base → C6 overrides → full stack ---
source "$REPO_ROOT/runs/configs/v2_base.env"
export GATED_ATTN=headwise
export EMBED_BITS=7
export EMBED_CLIP_SIGMAS=15.0
source "$REPO_ROOT/runs/configs/v2_fullstack.env"

export SEED=${SEED:-42}
export RUN_ID=${RUN_ID:-"fullstack_sb_ema990_pqttt"}
NGPUS="${NGPUS:-2}"

echo ""
echo "============================================"
echo "  FULL STACK: Small Batch + EMA=0.990 + PreQuantTTT"
echo "============================================"
echo "  RUN_ID:              $RUN_ID"
echo "  NGPUS:               $NGPUS"
echo "  SEED:                $SEED"
echo "  --- Small Batch ---"
echo "  GRAD_ACCUM_STEPS:    $GRAD_ACCUM_STEPS"
echo "  TRAIN_BATCH_TOKENS:  $TRAIN_BATCH_TOKENS"
echo "  --- EMA ---"
echo "  EMA_DECAY:           $EMA_DECAY"
echo "  --- PreQuantTTT ---"
echo "  PREQUANT_TTT_ENABLED: $PREQUANT_TTT_ENABLED"
echo "  PREQUANT_TTT_EPOCHS: $PREQUANT_TTT_EPOCHS"
echo "  PREQUANT_TTT_LR:     $PREQUANT_TTT_LR"
echo "  PREQUANT_TTT_LR_END: $PREQUANT_TTT_LR_END"
echo "  --- Other ---"
echo "  GATED_ATTN:          $GATED_ATTN"
echo "  EMBED_BITS:          $EMBED_BITS"
echo "  COMPRESSOR:          $COMPRESSOR"
echo "============================================"
echo ""

torchrun --standalone --nproc_per_node=$NGPUS train_gpt_v2.py
echo "==> Done. Log: logs/${RUN_ID}.txt"

# --- Results summary ---
echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"

python3 -c "
import os, re

log_file = 'logs/${RUN_ID}.txt'
if not os.path.isfile(log_file):
    print('  ERROR: log file not found')
    exit(1)

text = open(log_file).read()

# Extract metrics
metrics = {}
for label, pattern in [
    ('Pre-Q BPB',   r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)'),
    ('PostPQ BPB',  r'post-prequant-ttt val_loss:[\d.]+ val_bpb:([\d.]+)'),
    ('SW BPB',      r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)'),
    ('TTT BPB',     r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)'),
]:
    m = re.search(pattern, text)
    metrics[label] = float(m.group(1)) if m else None

m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
weights = int(m.group(1)) if m else None
m = re.search(r'model_params:(\d+)', text)
params = int(m.group(1)) if m else None

# Find last training step
steps = None
for m in re.finditer(r'step (\d+) \|', text):
    steps = int(m.group(1))

print()
print('  === Full Stack Results ===')
for k, v in metrics.items():
    print(f'  {k:>12}: {v:.4f}' if v else f'  {k:>12}: ?')
if weights: print(f'  {\"Weights\":>12}: {weights:,} ({weights/1e6:.2f} MB)')
if params:  print(f'  {\"Params\":>12}: {params:,}')
if steps:   print(f'  {\"Steps\":>12}: {steps}')

budget = 'YES' if weights and weights < 16_000_000 else 'NO/UNKNOWN'
print(f'  {\"Budget\":>12}: {budget}')

print()
print('  === Comparison ===')
print(f'  {\"\":>20} {\"TTT BPB\":>9}   Notes')
print(f'  {\"---\":>20} {\"---\":>9}   ---')
ttt = metrics.get('TTT BPB')
if ttt:
    print(f'  {\">> THIS RUN <<\":>20} {ttt:>9.4f}')
else:
    print(f'  {\">> THIS RUN <<\":>20} {\"?\":>9}')
print(f'  {\"R4 (EMA+PQ)\":>20} {\"1.0507\":>9}   EMA=0.990 + PreQuantTTT (no small batch)')
print(f'  {\"B2 (Small Batch)\":>20} {\"1.1419\":>9}   ga=1, batch÷4 (no EMA/PQ changes)')
print(f'  {\"R3 (EMA=0.990)\":>20} {\"1.1505\":>9}   EMA only (no small batch, no PQ)')
print(f'  {\"C6 (baseline)\":>20} {\"1.1622\":>9}   V2 headwise + emb7+eclip15')
print(f'  {\"C6 8xH100\":>20} {\"1.0805\":>9}   3-seed mean')
print(f'  {\"SOTA\":>20} {\"1.0136\":>9}   PR #1958 (okezue)')

# Pass/fail assessment
if ttt:
    print()
    if ttt < 1.00:
        print('  VERDICT: HOME RUN — techniques stack super-additively!')
        print('  ACTION: Run 8xH100 3-seed immediately.')
    elif ttt < 1.03:
        print('  VERDICT: GOOD — techniques stack roughly additively.')
        print('  ACTION: Run 8xH100 3-seed.')
    elif ttt < 1.05:
        print('  VERDICT: OK — partial stacking, some diminishing returns.')
        print('  ACTION: Investigate. Try EMA sweep with small batch.')
    else:
        print('  VERDICT: BAD — no improvement over R4 (1.0507).')
        print('  ACTION: Test without Small Batch. Small Batch may hurt PreQuantTTT.')
print()
"
