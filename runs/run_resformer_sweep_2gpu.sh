#!/bin/bash

# ResFormer (Value Residual Learning) Sweep — 2×H100
#
# Tests V₀ residual: v = (1-alpha)*v_current + alpha*v_layer0
# Paper: ResFormer (ACL 2025) — equivalent quality with 16% fewer params.
#
# Alpha sweep + config variants (8 runs):
#   R0: alpha=0.0 (control, skip if Q0 baseline reusable)  ~13 min
#   R1: alpha=0.1                                           ~13 min
#   R2: alpha=0.3                                           ~13 min
#   R3: alpha=0.5                                           ~13 min
#   R4: alpha=0.7                                           ~13 min
#   R5: best alpha, 11L                                     ~13 min
#   R6: best alpha, GQA (NUM_KV_HEADS=4)                    ~13 min
#   R7: best alpha, dim=448 headwise                        ~13 min
#
# Base config: dim=512, 10L, MHA, elementwise + GPTQ int7 + train data
# Usage (from repo root):  bash runs/run_resformer_sweep_2gpu.sh
# Expected time: ~104 min total (8 × ~13 min)

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/gptq_tune_10L_mha.env"

echo "============================================"
echo "  ResFormer (Value Residual) Sweep — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'VALUE_RESIDUAL_ALPHA env': 'VALUE_RESIDUAL_ALPHA' in code,
    'value_residual_alpha param': 'value_residual_alpha' in code,
    'vr_alpha in forward': 'vr_alpha' in code,
    'v0 blending': '(1 - vr_alpha) * v + vr_alpha * v0' in code,
}
all_ok = all(checks.values())
for name, ok in checks.items():
    print(f'  {chr(10003) if ok else chr(10007)} {name}')
print(f'\nPreflight: {\"PASS\" if all_ok else \"FAIL\"} ({sum(checks.values())}/{len(checks)})')
if not all_ok:
    print('ERROR: ResFormer code not found. Did you pull the latest branch?')
    sys.exit(1)
" || exit 1

# --- Helper ---
run_resformer() {
    local label="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "############################################################"
    echo ""

    source "$BASE_CONFIG"
    export RUN_ID="$run_id"
    export NGPUS=$NGPUS
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA MODEL_DIM=$MODEL_DIM NUM_LAYERS=$NUM_LAYERS NUM_KV_HEADS=$NUM_KV_HEADS GATED_ATTN=$GATED_ATTN"

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
}

# ==========================================================================
# Alpha Sweep (R0-R4): fix config, vary alpha
# ==========================================================================
run_resformer "R0: alpha=0.0 (control)" "resformer_a0" \
    "VALUE_RESIDUAL_ALPHA=0.0"
run_resformer "R1: alpha=0.1" "resformer_a01" \
    "VALUE_RESIDUAL_ALPHA=0.1"
run_resformer "R2: alpha=0.3" "resformer_a03" \
    "VALUE_RESIDUAL_ALPHA=0.3"
run_resformer "R3: alpha=0.5" "resformer_a05" \
    "VALUE_RESIDUAL_ALPHA=0.5"
run_resformer "R4: alpha=0.7" "resformer_a07" \
    "VALUE_RESIDUAL_ALPHA=0.7"

# ==========================================================================
# Config Variants (R5-R7): use best alpha (UPDATE AFTER ALPHA SWEEP)
# Default: alpha=0.3 (paper's range). Update BEST_ALPHA after R0-R4 results.
# ==========================================================================
BEST_ALPHA="${BEST_ALPHA:-0.3}"

run_resformer "R5: best alpha, 11L" "resformer_11L" \
    "VALUE_RESIDUAL_ALPHA=$BEST_ALPHA" "NUM_LAYERS=11"
run_resformer "R6: best alpha, GQA" "resformer_gqa" \
    "VALUE_RESIDUAL_ALPHA=$BEST_ALPHA" "NUM_KV_HEADS=4"
run_resformer "R7: best alpha, dim=448 headwise" "resformer_dim448" \
    "VALUE_RESIDUAL_ALPHA=$BEST_ALPHA" "MODEL_DIM=448" "GATED_ATTN=headwise"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================
echo ""
echo "############################################################"
echo "  RESFORMER SWEEP — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('R0', 'resformer_a0',    'alpha=0.0 (ctrl)'),
    ('R1', 'resformer_a01',   'alpha=0.1'),
    ('R2', 'resformer_a03',   'alpha=0.3'),
    ('R3', 'resformer_a05',   'alpha=0.5'),
    ('R4', 'resformer_a07',   'alpha=0.7'),
    ('R5', 'resformer_11L',   'best alpha, 11L'),
    ('R6', 'resformer_gqa',   'best alpha, GQA'),
    ('R7', 'resformer_dim448','best alpha, d448'),
]

pg_dir = '$PG_DIR'
na = '---'

print(f'  {\"Run\":>3} | {\"Config\":<20} | {\"Pre-Q BPB\":>9} | {\"TTT BPB\":>8} | {\"GPTQ Gap\":>8} | {\"Size\":>10} | {\"GPTQ t\":>6}')
print(f'  {\"-\"*3} | {\"-\"*20} | {\"-\"*9} | {\"-\"*8} | {\"-\"*8} | {\"-\"*10} | {\"-\"*6}')

for label, run_id, desc in runs:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<20} | {na:>9} | {na:>8} | {na:>8} | {na:>10} | {na:>6}')
        continue
    text = open(log_file).read()

    pre_bpb = '?'
    vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
    if vals: pre_bpb = vals[-1]

    ttt_bpb = '?'
    m = re.search(r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    gap = '?'
    try: gap = f'{float(ttt_bpb) - float(pre_bpb):+.4f}'
    except: pass

    sz = '?'
    m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', text)
    if m: sz = f'{int(m.group(1))/1e6:.2f}MB'

    gptq_t = '?'
    m = re.search(r'gptq:done in ([\d.]+)s', text)
    if m: gptq_t = f'{float(m.group(1)):.0f}s'

    print(f'  {label:>3} | {desc:<20} | {pre_bpb:>9} | {ttt_bpb:>8} | {gap:>8} | {sz:>10} | {gptq_t:>6}')

print()
print('  Q0 baseline (no ResFormer): pre-Q 1.2035, TTT 1.2579, gap +0.0544')
print('  Success = any alpha produces lower pre-Q BPB than 1.2035')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
