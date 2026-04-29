#!/bin/bash

# GPTQ Validation — 2×H100 RunPod
#
# Tests full GPTQ (Frantar et al., ICLR 2023) ported from PG rank 9.
# Two runs: smoke test (known-good config) then primary target (elementwise dim=512).
#
# Usage (from repo root):
#   bash runs/run_gptq_2gpu.sh

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

echo "============================================"
echo "  GPTQ Validation — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight: verify GPTQ code is present ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'gptq_quantize_weight': 'gptq_quantize_weight' in code,
    '_save_gptq hooks': '_save_gptq' in code,
    '_byte_shuffle': '_byte_shuffle' in code,
    'generate_autoregressive_calib': 'generate_autoregressive_calib' in code,
    'USE_GPTQ env var': 'USE_GPTQ' in code,
}
all_ok = all(checks.values())
for name, ok in checks.items():
    print(f'  {chr(10003) if ok else chr(10007)} {name}')
print(f'\nPreflight: {\"PASS\" if all_ok else \"FAIL\"} ({sum(checks.values())}/{len(checks)})')
if not all_ok:
    print('ERROR: GPTQ code not found. Did you pull the latest branch?')
    sys.exit(1)
" || exit 1

# ==========================================================================
# SMOKE TEST — Headwise dim=448 + GPTQ (known-good config)
# Compare to Run A (1.2411 on 2×H100, int8+zlib)
# PASS: no crash, BPB gap < 0.02 vs int8, GPTQ time < 60s
# ==========================================================================
echo ""
echo "############################################################"
echo "  SMOKE: Headwise dim=448 + GPTQ int6"
echo "  GPUs: $NGPUS | Compare to Run A (1.2411, int8)"
echo "############################################################"
echo ""

source "$REPO_ROOT/runs/configs/sp8192_combo_slim_gptq.env"
echo "VERIFY: USE_GPTQ=$USE_GPTQ QUANT_MODE=$QUANT_MODE GATED_ATTN=$GATED_ATTN MODEL_DIM=$MODEL_DIM"
export NGPUS=$NGPUS
export RUN_ID=sp8192_combo_slim_gptq_2gpu
SMOKE_OK=1
bash "$REPO_ROOT/runs/parameter_golf_baseline.sh" || SMOKE_OK=0

# --- Auto-check smoke test results ---
SMOKE_LOG="$PG_DIR/logs/sp8192_combo_slim_gptq_2gpu.txt"
echo ""
echo "=== SMOKE TEST AUTO-CHECK ==="
if [ "$SMOKE_OK" -eq 0 ]; then
    echo "  FAIL: training script crashed"
    echo "  Check: $SMOKE_LOG"
    echo ""
    echo "Aborting — fix smoke test before running main experiment."
    exit 1
fi

python3 -c "
import re, sys

log = open('$SMOKE_LOG').read()
passed = 0
total = 4

# 1. GPTQ calibration completed
m = re.search(r'gptq:done in ([\d.]+)s', log)
if m:
    t = float(m.group(1))
    ok = t < 120
    print(f'  {chr(10003) if ok else chr(10007)} GPTQ calibration: {t:.1f}s {\"(< 120s)\" if ok else \"(TOO SLOW)\"}')
    if ok: passed += 1
else:
    print(f'  {chr(10007)} GPTQ calibration: NOT FOUND')

# 2. BPB gap vs int8 baseline (Run A = 1.2411)
bpb = None
for pat in [r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
            r'final_int6_brotli_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)']:
    m = re.search(pat, log)
    if m:
        bpb = float(m.group(1))
        break
if bpb:
    gap = bpb - 1.2411
    ok = gap < 0.02
    print(f'  {chr(10003) if ok else chr(10007)} BPB: {bpb:.4f} (gap vs Run A: {gap:+.4f}, need < +0.02)')
    if ok: passed += 1
else:
    print(f'  {chr(10007)} BPB: NOT FOUND')

# 3. Artifact under 16 MB
m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', log)
if m:
    sz = int(m.group(1))
    mb = sz / 1e6
    ok = sz < 16_000_000
    print(f'  {chr(10003) if ok else chr(10007)} Size: {mb:.2f} MB {\"(under 16 MB)\" if ok else \"(OVER BUDGET)\"}')
    if ok: passed += 1
else:
    print(f'  {chr(10007)} Size: NOT FOUND')

# 4. GPTQ Hessians collected
m = re.search(r'gptq:done in [\d.]+s, (\d+) Hessians collected', log)
if m:
    n_hess = int(m.group(1))
    ok = n_hess > 0
    print(f'  {chr(10003) if ok else chr(10007)} GPTQ Hessians: {n_hess} collected')
    if ok: passed += 1
else:
    print(f'  {chr(10007)} GPTQ Hessians: NOT FOUND')

print(f'\nSmoke test: {\"PASS\" if passed == total else \"FAIL\"} ({passed}/{total})')
if passed < total:
    print('Aborting — fix smoke test issues before running main experiment.')
    sys.exit(1)
" || exit 1

# ==========================================================================
# MAIN — Elementwise dim=512 + GPTQ (primary GPTQ target)
# Full dim=512 with Hessian-based error compensation.
# Expected ~10 MB artifact, BPB gap ~+0.01 vs float.
# ==========================================================================
echo ""
echo "############################################################"
echo "  MAIN: Elementwise dim=512 + GPTQ int6"
echo "  GPUs: $NGPUS | PRIMARY GPTQ TARGET"
echo "############################################################"
echo ""

source "$REPO_ROOT/runs/configs/sp8192_elementwise_gptq_dim512.env"
echo "VERIFY: USE_GPTQ=$USE_GPTQ QUANT_MODE=$QUANT_MODE GATED_ATTN=$GATED_ATTN MODEL_DIM=$MODEL_DIM"
export NGPUS=$NGPUS
export RUN_ID=sp8192_elem_gptq_dim512_2gpu
bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================
echo ""
echo "############################################################"
echo "  GPTQ VALIDATION — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('SMOKE', 'sp8192_combo_slim_gptq_2gpu',  'Headwise dim=448 GPTQ', '1.2411'),
    ('MAIN',  'sp8192_elem_gptq_dim512_2gpu', 'Elem dim=512 GPTQ',     '1.2338'),
]

pg_dir = '$PG_DIR'

print(f'  {\"Run\":>5} | {\"Config\":<22} | {\"BPB\":>8} | {\"Ref\":>8} | {\"Gap\":>8} | {\"Size\":>10} | Budget | GPTQ time')
print(f'  {\"-\"*5} | {\"-\"*22} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*10} | ------ | ---------')

for label, run_id, desc, ref_bpb in runs:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>5} | {desc:<22} | {\"---\":>8} | {ref_bpb:>8} | {\"---\":>8} | {\"---\":>10} | ---    | ---')
        continue
    text = open(log_file).read()

    bpb = '?'
    for pat in [r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)',
                r'final_int6_brotli_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)']:
        m = re.search(pat, text)
        if m: bpb = m.group(1); break

    comp, budget = '?', '?'
    m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', text)
    if m:
        sz = int(m.group(1))
        comp = f'{sz/1e6:.2f}MB'
        budget = 'YES' if sz < 16_000_000 else 'NO'

    gptq_time = '?'
    m = re.search(r'gptq:done in ([\d.]+)s', text)
    if m: gptq_time = f'{float(m.group(1)):.0f}s'

    gap = '?'
    try: gap = f'{float(bpb) - float(ref_bpb):+.4f}'
    except: pass

    print(f'  {label:>5} | {desc:<22} | {bpb:>8} | {ref_bpb:>8} | {gap:>8} | {comp:>10} | {budget:<6} | {gptq_time}')

print()
print('  References:')
print('    Run A  (headwise dim=448, int8+zlib) = 1.2411 BPB, 15.03 MB')
print('    Run E1 (elem dim=448, int8+zlib)     = 1.2338 BPB, 16.67 MB (OVER)')
print('    Run 11 (8xH100, int8+zlib)           = 1.2077 BPB, 15.35 MB')
print()
print('  PASS = GPTQ gap < 0.02 AND size < 16 MB')
print('  If PASS -> run on 8xH100 for final submission')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
