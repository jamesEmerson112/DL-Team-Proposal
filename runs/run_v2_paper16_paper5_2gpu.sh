#!/bin/bash
# =============================================================================
# Paper #16 (LR Warmup) + Paper #5 (Structured FFN) — 2xH100
#
# 6 runs: R0 (C6 control) + R1-R3 (LR warmup sweep) + R4-R5 (Structured FFN)
# All independent experiments against C6 baseline.
#
# Usage (from parameter-golf/ directory):
#   bash ../runs/run_v2_paper16_paper5_2gpu.sh
#
# Estimated time: ~78 min (6 runs x ~13 min each)
# =============================================================================

set -uo pipefail

NGPUS="${NGPUS:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/v2_base.env"
TRAIN_SCRIPT="train_gpt_v2.py"

echo "============================================"
echo "  Paper #16 + #5: LR Warmup + Structured FFN"
echo "  $(date)"
echo "  GPUs: $NGPUS"
echo "============================================"

# --- Preflight ---
cd "$PG_DIR"
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found. Install FA3 wheel first."
    exit 1
}
python3 -c "import brotli" 2>/dev/null || {
    echo "ERROR: brotli not found. pip install brotli"
    exit 1
}
echo "Preflight: FA3 + brotli OK"
mkdir -p logs

# --- Helper ---
run_v2() {
    local label="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "  $(date)"
    echo "############################################################"
    echo ""

    source "$BASE_CONFIG"
    export RUN_ID="$run_id"
    # C6 overrides (always applied as base)
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    # Experiment-specific overrides (AFTER base + C6)
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: GATED_ATTN=$GATED_ATTN LR_WARMUP_FRAC=$LR_WARMUP_FRAC STRUCTURED_FFN=$STRUCTURED_FFN FFN_RANK_RATIO=$FFN_RANK_RATIO FFN_NUM_BLOCKS=$FFN_NUM_BLOCKS RUN_ID=$RUN_ID"

    if torchrun --standalone --nproc_per_node=$NGPUS "$TRAIN_SCRIPT" 2>&1 | tee "logs/${run_id}.txt"; then
        echo "==> $label: OK"
    else
        echo "==> $label: FAILED (exit code $?, check logs/$run_id.txt)"
    fi
}

# ==========================================
# RUN 0: C6 BASELINE (CONTROL)
# ==========================================
run_v2 "R0: C6 Baseline (control)" "v2_p16p5_r0_baseline"

# ==========================================
# RUNS 1-3: LR WARMUP SWEEP (Paper #16)
# ==========================================
run_v2 "R1: LR Warmup 2%" "v2_p16p5_r1_warmup002" \
    "LR_WARMUP_FRAC=0.02"

run_v2 "R2: LR Warmup 5%" "v2_p16p5_r2_warmup005" \
    "LR_WARMUP_FRAC=0.05"

run_v2 "R3: LR Warmup 10%" "v2_p16p5_r3_warmup010" \
    "LR_WARMUP_FRAC=0.10"

# ==========================================
# RUNS 4-5: STRUCTURED FFN (Paper #5)
# ==========================================
run_v2 "R4: Structured FFN r=0.5 b=4" "v2_p16p5_r4_sffn_r50_b4" \
    "STRUCTURED_FFN=1" "FFN_RANK_RATIO=0.5" "FFN_NUM_BLOCKS=4"

run_v2 "R5: Structured FFN r=0.75 b=8" "v2_p16p5_r5_sffn_r75_b8" \
    "STRUCTURED_FFN=1" "FFN_RANK_RATIO=0.75" "FFN_NUM_BLOCKS=8"

# ==========================================
# RESULTS SUMMARY
# ==========================================

echo ""
echo "############################################################"
echo "  ALL 6 RUNS COMPLETE — RESULTS SUMMARY"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('R0', 'v2_p16p5_r0_baseline',    'C6 Baseline (control)'),
    ('R1', 'v2_p16p5_r1_warmup002',   'LR Warmup 2%'),
    ('R2', 'v2_p16p5_r2_warmup005',   'LR Warmup 5%'),
    ('R3', 'v2_p16p5_r3_warmup010',   'LR Warmup 10%'),
    ('R4', 'v2_p16p5_r4_sffn_r50_b4', 'Struct FFN r50 b4'),
    ('R5', 'v2_p16p5_r5_sffn_r75_b8', 'Struct FFN r75 b8'),
]

print(f\"  {'Run':>3} | {'Config':<22} | {'Params':>8} | {'Pre-Q BPB':>9} | {'TTT BPB':>8} | {'Size':>10} | {'Steps':>5}\")
print(f\"  {'-'*3} | {'-'*22} | {'-'*8} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*5}\")

for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<22} | {\"---\":>8} | {\"---\":>9} | {\"---\":>8} | {\"---\":>10} | {\"---\":>5}')
        continue
    text = open(log_file).read()

    params = '?'
    m = re.search(r'model_params:(\d+)', text)
    if m: params = f'{int(m.group(1))/1e6:.1f}M'

    pre_bpb = '?'
    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: pre_bpb = m.group(1)

    ttt_bpb = '?'
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    sz = '?'
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: sz = f'{int(m.group(1))/1e6:.2f}MB'

    steps = '?'
    m = re.search(r'stopping_early.*?step: (\d+)', text)
    if not m:
        vals = re.findall(r'(\d+)/\d+ val_loss:', text)
        if vals: steps = vals[-1]

    print(f'  {label:>3} | {desc:<22} | {params:>8} | {pre_bpb:>9} | {ttt_bpb:>8} | {sz:>10} | {steps:>5}')

print()
print(f'  C6 reference (2xH100): 1.1622 TTT BPB')
print(f'  C6 reference (8xH100): 1.0805 TTT BPB')
print()

# Highlight winners
bpb_results = {}
for label, run_id, desc in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file): continue
    text = open(log_file).read()
    m = re.search(r'quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: bpb_results[label] = (float(m.group(1)), desc)

if 'R0' in bpb_results:
    baseline = bpb_results['R0'][0]
    print(f'  === vs R0 baseline ({baseline:.4f}) ===')
    for label in ['R1','R2','R3','R4','R5']:
        if label in bpb_results:
            bpb, desc = bpb_results[label]
            delta = bpb - baseline
            verdict = 'BETTER' if delta < -0.001 else 'WORSE' if delta > 0.001 else 'NEUTRAL'
            print(f'  {label} ({desc}): {bpb:.4f} ({delta:+.4f}) — {verdict}')
"

echo ""
echo "============================================"
echo "  Done. All 6 runs complete."
echo "  Logs in parameter-golf/logs/"
echo "============================================"
