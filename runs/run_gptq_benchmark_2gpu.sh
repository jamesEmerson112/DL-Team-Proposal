#!/bin/bash

# GPTQ Quantization Benchmark — 2×H100 RunPod
#
# Tests 4 GPTQ configurations to find the best match for our train_gpt.py:
#   G1: int7 (clip=63) + k×std + AR self-gen calibration
#   G2: int7 (clip=63) + k×std + training data calibration
#   G3: int6 (clip=31) + k×std + AR self-gen calibration
#   G4: int6 (clip=31) + k×std + training data calibration
#
# All use headwise dim=448 (sp8192_combo_slim base config) for controlled comparison.
# Reference: Run A (int8+zlib, no GPTQ) = 1.2411 BPB, 15.03 MB
# Previous GPTQ run (5-percentile, int6, AR): 1.2929 BPB (TTT), 10.5 MB — gap +0.0518
#
# Changes from previous GPTQ:
#   1. inference_mode → no_grad (fixes tensor poisoning)
#   2. 5-percentile search → k×std single pass (Kevin Clark rank 5)
#   3. int7 (clip_range=63) option (Kevin Clark rank 5)
#   4. Training data calibration option (Kevin Clark rank 5)
#
# Usage (from repo root):  bash runs/run_gptq_benchmark_2gpu.sh
# Expected time: ~80-90 min total (4 runs × ~20 min each)

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

echo "============================================"
echo "  GPTQ Quantization Benchmark — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'no_grad (not inference_mode)': 'torch.no_grad()' in code and 'gptq' in code.split('torch.no_grad()')[0][-200:],
    'clip_sigmas param': 'clip_sigmas' in code,
    'GPTQ_CLIP_RANGE env': 'GPTQ_CLIP_RANGE' in code,
    'GPTQ_CALIB_SOURCE env': 'GPTQ_CALIB_SOURCE' in code,
}
all_ok = all(checks.values())
for name, ok in checks.items():
    print(f'  {chr(10003) if ok else chr(10007)} {name}')
print(f'\nPreflight: {\"PASS\" if all_ok else \"FAIL\"} ({sum(checks.values())}/{len(checks)})')
if not all_ok:
    print('ERROR: Updated GPTQ code not found. Did you pull the latest branch?')
    sys.exit(1)
" || exit 1

# Base config: sp8192_combo_slim (headwise dim=448 + TTT)
BASE_CONFIG="$REPO_ROOT/runs/configs/sp8192_combo_slim.env"

run_gptq_variant() {
    local label="$1"
    local run_id="$2"
    local clip_range="$3"
    local calib_source="$4"

    echo ""
    echo "############################################################"
    echo "  $label"
    echo "  clip_range=$clip_range calib=$calib_source | GPUs: $NGPUS"
    echo "############################################################"
    echo ""

    # Start from base config
    source "$BASE_CONFIG"
    # Override for GPTQ
    export QUANT_MODE=int6_brotli
    export SDCLIP_K=12.85
    export EMBED_CLIP_K=20.0
    export USE_GPTQ=1
    export GPTQ_AR_SAMPLES=64
    export GPTQ_AR_SEQLEN=2048
    export GPTQ_TEMPERATURE=0.8
    export GPTQ_BATCH_SIZE=8
    export GPTQ_RESERVE_MS=0
    export GPTQ_CLIP_RANGE="$clip_range"
    export GPTQ_CALIB_SOURCE="$calib_source"
    export NGPUS=$NGPUS
    export RUN_ID="$run_id"

    echo "VERIFY: USE_GPTQ=$USE_GPTQ QUANT_MODE=$QUANT_MODE GPTQ_CLIP_RANGE=$GPTQ_CLIP_RANGE GPTQ_CALIB_SOURCE=$GPTQ_CALIB_SOURCE"

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        echo "$label: OK"
    else
        echo "$label: FAILED (check logs/$run_id.txt)"
    fi
}

# ==========================================================================
# G1: int7 + AR self-gen (Kevin Clark's bitwidth + rank 9's calibration)
# ==========================================================================
run_gptq_variant \
    "G1: int7 (clip=63) + AR self-gen" \
    "gptq_bench_int7_ar" \
    63 ar

# ==========================================================================
# G2: int7 + training data (Kevin Clark's full approach)
# ==========================================================================
run_gptq_variant \
    "G2: int7 (clip=63) + training data" \
    "gptq_bench_int7_train" \
    63 train

# ==========================================================================
# G3: int6 + AR self-gen (rank 9's bitwidth + rank 9's calibration)
# ==========================================================================
run_gptq_variant \
    "G3: int6 (clip=31) + AR self-gen" \
    "gptq_bench_int6_ar" \
    31 ar

# ==========================================================================
# G4: int6 + training data (rank 9's bitwidth + Kevin Clark's calibration)
# ==========================================================================
run_gptq_variant \
    "G4: int6 (clip=31) + training data" \
    "gptq_bench_int6_train" \
    31 train

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================
echo ""
echo "############################################################"
echo "  GPTQ BENCHMARK — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('G1', 'gptq_bench_int7_ar',    'int7 + AR self-gen',   63, 'ar'),
    ('G2', 'gptq_bench_int7_train', 'int7 + train data',    63, 'train'),
    ('G3', 'gptq_bench_int6_ar',    'int6 + AR self-gen',   31, 'ar'),
    ('G4', 'gptq_bench_int6_train', 'int6 + train data',    31, 'train'),
]

pg_dir = '$PG_DIR'
ref_bpb = 1.2411  # Run A (int8, no GPTQ)

print(f'  {\"Run\":>3} | {\"Config\":<20} | {\"Pre-Q BPB\":>9} | {\"RT BPB\":>8} | {\"TTT BPB\":>8} | {\"Gap\":>7} | {\"Size\":>10} | {\"GPTQ t\":>6}')
print(f'  {\"-\"*3} | {\"-\"*20} | {\"-\"*9} | {\"-\"*8} | {\"-\"*8} | {\"-\"*7} | {\"-\"*10} | {\"-\"*6}')

for label, run_id, desc, clip, calib in runs:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<20} | {\"---\":>9} | {\"---\":>8} | {\"---\":>8} | {\"---\":>7} | {\"---\":>10} | {\"---\":>6}')
        continue
    text = open(log_file).read()

    # Pre-quant BPB (last training val)
    pre_bpb = '?'
    vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
    if vals: pre_bpb = vals[-1]

    # Roundtrip BPB
    rt_bpb = '?'
    m = re.search(r'final_int6_brotli_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: rt_bpb = m.group(1)

    # TTT BPB
    ttt_bpb = '?'
    m = re.search(r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)

    # Size
    comp, sz_str = '?', '?'
    m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', text)
    if m:
        sz = int(m.group(1))
        comp = f'{sz/1e6:.2f}MB'

    # GPTQ time
    gptq_t = '?'
    m = re.search(r'gptq:done in ([\d.]+)s', text)
    if m: gptq_t = f'{float(m.group(1)):.0f}s'

    # Gap vs Run A
    gap = '?'
    best = ttt_bpb if ttt_bpb != '?' else rt_bpb
    try: gap = f'{float(best) - ref_bpb:+.4f}'
    except: pass

    print(f'  {label:>3} | {desc:<20} | {pre_bpb:>9} | {rt_bpb:>8} | {ttt_bpb:>8} | {gap:>7} | {comp:>10} | {gptq_t:>6}')

print()
print('  References:')
print('    Run A  (int8+zlib, no GPTQ)          = 1.2411 BPB, 15.03 MB')
print('    Prev GPTQ (5-pct, int6, AR, old code) = 1.2929 BPB (TTT), 10.50 MB, gap +0.0518')
print('    Kevin Clark rank 5 (int7, train, GPTQ) = ~0.012 BPB gap')
print()
print('  Winner = lowest TTT BPB that fits under 16 MB')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
