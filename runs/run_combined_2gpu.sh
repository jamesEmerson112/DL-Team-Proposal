#!/bin/bash

# Combined Run — GPTQ Tuning + ResFormer Sweep (2×H100)
#
# Part A: GPTQ Quality Tuning (3 runs, ~63 min)
#   Q1: Sequential block-by-block quantization      ~25 min
#   Q3: GPTQ on embedding layer                     ~13 min
#   Q7: All combined (seq + hooks + embed)          ~25 min
#
# Part B: ResFormer Alpha Sweep (5 runs, ~65 min)
#   R0: alpha=0.0 (control)                          ~13 min
#   R1: alpha=0.1                                    ~13 min
#   R2: alpha=0.3                                    ~13 min
#   R3: alpha=0.5                                    ~13 min
#   R4: alpha=0.7                                    ~13 min
#
# Total: 8 runs, ~128 min (~2h8m)
# Config variants (R5-R7) deferred — need best alpha from Part B first.
#
# Base config: dim=512, 10L, MHA, elementwise + GPTQ int7 + train data
# Usage (from repo root):  bash runs/run_combined_2gpu.sh

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/gptq_tune_10L_mha.env"

echo "============================================"
echo "  Combined: GPTQ Tuning + ResFormer — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'GPTQ_SEQUENTIAL': 'GPTQ_SEQUENTIAL' in code,
    'GPTQ_USE_HOOKS': 'GPTQ_USE_HOOKS' in code,
    'GPTQ_EMBED': 'GPTQ_EMBED' in code,
    'gptq_sequential_quantize': 'gptq_sequential_quantize' in code,
    'VALUE_RESIDUAL_ALPHA': 'VALUE_RESIDUAL_ALPHA' in code,
    'vr_alpha blending': 'vr_alpha' in code,
}
all_ok = all(checks.values())
for name, ok in checks.items():
    print(f'  {chr(10003) if ok else chr(10007)} {name}')
print(f'\nPreflight: {\"PASS\" if all_ok else \"FAIL\"} ({sum(checks.values())}/{len(checks)})')
if not all_ok:
    print('ERROR: Required code not found. Did you pull the latest branch?')
    sys.exit(1)
" || exit 1

# --- Helper ---
run_experiment() {
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

    echo "VERIFY: GPTQ_SEQUENTIAL=$GPTQ_SEQUENTIAL GPTQ_USE_HOOKS=$GPTQ_USE_HOOKS GPTQ_EMBED=$GPTQ_EMBED VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA"

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
}

# ==========================================
# PART A: GPTQ Quality Tuning (Q0 already done)
# ==========================================
echo ""
echo "========== PART A: GPTQ TUNING =========="
echo ""

run_experiment "Q1: Sequential block quantization" "gptq_tune_sequential" \
    "GPTQ_SEQUENTIAL=1"
run_experiment "Q3: GPTQ on embeddings" "gptq_tune_embed" \
    "GPTQ_EMBED=1"
run_experiment "Q7: All GPTQ combined" "gptq_tune_all" \
    "GPTQ_SEQUENTIAL=1" "GPTQ_USE_HOOKS=1" "GPTQ_EMBED=1"

# ==========================================
# PART B: ResFormer Alpha Sweep
# ==========================================
echo ""
echo "========== PART B: RESFORMER SWEEP =========="
echo ""

run_experiment "R0: alpha=0.0 (control)" "resformer_a0" \
    "VALUE_RESIDUAL_ALPHA=0.0"
run_experiment "R1: alpha=0.1" "resformer_a01" \
    "VALUE_RESIDUAL_ALPHA=0.1"
run_experiment "R2: alpha=0.3" "resformer_a03" \
    "VALUE_RESIDUAL_ALPHA=0.3"
run_experiment "R3: alpha=0.5" "resformer_a05" \
    "VALUE_RESIDUAL_ALPHA=0.5"
run_experiment "R4: alpha=0.7" "resformer_a07" \
    "VALUE_RESIDUAL_ALPHA=0.7"

# ==========================================
# RESULTS SUMMARY
# ==========================================
echo ""
echo "############################################################"
echo "  COMBINED RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('', '', '--- GPTQ TUNING ---'),
    ('Q0', 'gptq_tune_baseline',   'Baseline (done)'),
    ('Q1', 'gptq_tune_sequential', 'Sequential blocks'),
    ('Q3', 'gptq_tune_embed',      'GPTQ embeddings'),
    ('Q7', 'gptq_tune_all',        'All GPTQ combined'),
    ('', '', '--- RESFORMER ---'),
    ('R0', 'resformer_a0',    'alpha=0.0 (ctrl)'),
    ('R1', 'resformer_a01',   'alpha=0.1'),
    ('R2', 'resformer_a03',   'alpha=0.3'),
    ('R3', 'resformer_a05',   'alpha=0.5'),
    ('R4', 'resformer_a07',   'alpha=0.7'),
]

pg_dir = '$PG_DIR'
na = '---'

print(f'  {\"Run\":>3} | {\"Config\":<20} | {\"Pre-Q BPB\":>9} | {\"TTT BPB\":>8} | {\"GPTQ Gap\":>8} | {\"Size\":>10} | {\"GPTQ t\":>6}')
print(f'  {\"-\"*3} | {\"-\"*20} | {\"-\"*9} | {\"-\"*8} | {\"-\"*8} | {\"-\"*10} | {\"-\"*6}')

for label, run_id, desc in runs:
    if not run_id:
        print(f'  {desc}')
        continue
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
print('  Q0 baseline: pre-Q 1.2035, TTT 1.2579, gap +0.0544')
print('  GPTQ target: gap < 0.03 | ResFormer target: pre-Q < 1.2035')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
