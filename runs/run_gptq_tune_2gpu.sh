#!/bin/bash

# GPTQ Quality Tuning — Close the Gap to Kevin Clark
#
# Tests 4 GPTQ improvements to reduce gap from ~0.05 to ~0.012 BPB:
#   1. Sequential block-by-block quantization (GPTQ_SEQUENTIAL)
#   2. register_forward_hook for Hessian collection (GPTQ_USE_HOOKS)
#   3. GPTQ on embedding layer (GPTQ_EMBED)
#   4. Dampening factor tuning (GPTQ_PERCDAMP)
#
# Base config: dim=512, 10L, MHA, elementwise + GPTQ int7 + train data
# Usage (from repo root):  bash runs/run_gptq_tune_2gpu.sh
# Expected time: ~104 min total (8 runs × ~13 min each)

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
BASE_CONFIG="$REPO_ROOT/runs/configs/gptq_tune_10L_mha.env"

echo "============================================"
echo "  GPTQ Quality Tuning — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'GPTQ_SEQUENTIAL env': 'GPTQ_SEQUENTIAL' in code,
    'GPTQ_USE_HOOKS env': 'GPTQ_USE_HOOKS' in code,
    'GPTQ_EMBED env': 'GPTQ_EMBED' in code,
    'GPTQ_PERCDAMP env': 'GPTQ_PERCDAMP' in code,
    'gptq_sequential_quantize fn': 'gptq_sequential_quantize' in code,
    'gptq_collect_hessians_with_hooks fn': 'gptq_collect_hessians_with_hooks' in code,
}
all_ok = all(checks.values())
for name, ok in checks.items():
    print(f'  {chr(10003) if ok else chr(10007)} {name}')
print(f'\nPreflight: {\"PASS\" if all_ok else \"FAIL\"} ({sum(checks.values())}/{len(checks)})')
if not all_ok:
    print('ERROR: GPTQ tuning code not found. Did you pull the latest branch?')
    sys.exit(1)
" || exit 1

# --- Helper ---
run_tune() {
    local label="$1"
    local run_id="$2"
    shift 2
    # Remaining args are env var overrides

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "############################################################"
    echo ""

    source "$BASE_CONFIG"
    export RUN_ID="$run_id"
    export NGPUS=$NGPUS
    # Apply overrides
    for override in "$@"; do
        export "$override"
    done

    echo "VERIFY: GPTQ_SEQUENTIAL=$GPTQ_SEQUENTIAL GPTQ_USE_HOOKS=$GPTQ_USE_HOOKS GPTQ_EMBED=$GPTQ_EMBED GPTQ_PERCDAMP=$GPTQ_PERCDAMP"

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
}

# ==========================================================================
# Q0: Baseline (current GPTQ on new 10L MHA config)
# ==========================================================================
run_tune "Q0: Baseline (current GPTQ)" "gptq_tune_baseline"

# ==========================================================================
# Q1: Sequential block-by-block quantization (biggest suspected fix)
# ==========================================================================
run_tune "Q1: Sequential block quantization" "gptq_tune_sequential" \
    "GPTQ_SEQUENTIAL=1"

# ==========================================================================
# Q2: Hook-based Hessian collection
# ==========================================================================
run_tune "Q2: register_forward_hook" "gptq_tune_hooks" \
    "GPTQ_USE_HOOKS=1"

# ==========================================================================
# Q3: GPTQ on embeddings
# ==========================================================================
run_tune "Q3: GPTQ on embeddings" "gptq_tune_embed" \
    "GPTQ_EMBED=1"

# ==========================================================================
# Q4: Lower dampening (0.001)
# ==========================================================================
run_tune "Q4: Dampening 0.001" "gptq_tune_damp001" \
    "GPTQ_PERCDAMP=0.001"

# ==========================================================================
# Q5: Higher dampening (0.05)
# ==========================================================================
run_tune "Q5: Dampening 0.05" "gptq_tune_damp05" \
    "GPTQ_PERCDAMP=0.05"

# ==========================================================================
# Q6: Sequential + Hooks combo
# ==========================================================================
run_tune "Q6: Sequential + Hooks" "gptq_tune_seq_hooks" \
    "GPTQ_SEQUENTIAL=1" "GPTQ_USE_HOOKS=1"

# ==========================================================================
# Q7: All combined (sequential + hooks + embed + best dampening TBD)
# ==========================================================================
run_tune "Q7: All combined" "gptq_tune_all" \
    "GPTQ_SEQUENTIAL=1" "GPTQ_USE_HOOKS=1" "GPTQ_EMBED=1"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================
echo ""
echo "############################################################"
echo "  GPTQ QUALITY TUNING — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('Q0', 'gptq_tune_baseline',   'Baseline (current)'),
    ('Q1', 'gptq_tune_sequential', 'Sequential blocks'),
    ('Q2', 'gptq_tune_hooks',      'forward_hook'),
    ('Q3', 'gptq_tune_embed',      'GPTQ embeddings'),
    ('Q4', 'gptq_tune_damp001',    'percdamp=0.001'),
    ('Q5', 'gptq_tune_damp05',     'percdamp=0.05'),
    ('Q6', 'gptq_tune_seq_hooks',  'Sequential+hooks'),
    ('Q7', 'gptq_tune_all',        'All combined'),
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

    # Pre-quant BPB
    pre_bpb = '?'
    vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
    if vals: pre_bpb = vals[-1]

    # TTT BPB
    ttt_bpb = '?'
    m = re.search(r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    # GPTQ gap
    gap = '?'
    try: gap = f'{float(ttt_bpb) - float(pre_bpb):+.4f}'
    except: pass

    # Size
    sz = '?'
    m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', text)
    if m: sz = f'{int(m.group(1))/1e6:.2f}MB'

    # GPTQ time
    gptq_t = '?'
    m = re.search(r'gptq:done in ([\d.]+)s', text)
    if m: gptq_t = f'{float(m.group(1)):.0f}s'

    print(f'  {label:>3} | {desc:<20} | {pre_bpb:>9} | {ttt_bpb:>8} | {gap:>8} | {sz:>10} | {gptq_t:>6}')

print()
print('  Target: GPTQ gap < 0.03 (halfway to Kevin Clark\\'s 0.012)')
print('  Current gap: ~0.05 (from benchmark sweep)')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
