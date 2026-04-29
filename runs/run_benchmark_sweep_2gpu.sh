#!/bin/bash

# Benchmark Sweep — Dim / Layers / Attention (2×H100, GPTQ)
#
# Three isolated sweeps, each varying one axis:
#   Sweep 1 (Dim):    448, 512, 768, 1024  — fix: 9L, GQA 8Q/4KV
#   Sweep 2 (Layers): 9, 10, 11            — fix: dim=512, GQA 8Q/4KV
#   Sweep 3 (Attn):   GQA, MHA             — fix: dim=512, 9L
#
# D2 (dim=512, 9L, GQA) is shared baseline across all 3 sweeps.
# All runs: elementwise gated attn + GPTQ int7 (clip=63) + train data calib.
#
# Usage (from repo root):  bash runs/run_benchmark_sweep_2gpu.sh
# Expected time: ~91 min total (7 runs × ~13 min each)

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"
CONFIG_DIR="$REPO_ROOT/runs/configs"

echo "============================================"
echo "  Benchmark Sweep — Dim / Layers / Attention"
echo "  2×H100 | GPTQ int7 + train data"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# --- Preflight ---
python3 -c "
import sys
with open('parameter-golf/train_gpt.py') as f:
    code = f.read()
checks = {
    'no_grad (not inference_mode)': 'torch.no_grad()' in code and 'gptq_collect_hessians' in code,
    'clip_sigmas param': 'clip_sigmas' in code,
    'GPTQ_CLIP_RANGE env': 'GPTQ_CLIP_RANGE' in code,
    'GPTQ_CALIB_SOURCE env': 'GPTQ_CALIB_SOURCE' in code,
    'elementwise gated attn': 'elementwise' in code,
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
RESULTS=()

run_bench() {
    local label="$1"
    local sweep="$2"
    local config_file="$3"

    echo ""
    echo "############################################################"
    echo "  $label  [$sweep]"
    echo "############################################################"
    echo ""

    source "$CONFIG_DIR/$config_file"
    export NGPUS=$NGPUS

    echo "VERIFY: MODEL_DIM=$MODEL_DIM NUM_LAYERS=$NUM_LAYERS NUM_KV_HEADS=$NUM_KV_HEADS GATED_ATTN=$GATED_ATTN USE_GPTQ=$USE_GPTQ GPTQ_CALIB_SOURCE=$GPTQ_CALIB_SOURCE"

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        echo "$label: OK"
        RESULTS+=("$label|$sweep|$config_file|OK")
    else
        echo "$label: FAILED (check logs/$RUN_ID.txt)"
        RESULTS+=("$label|$sweep|$config_file|FAIL")
    fi
}

# ==========================================================================
# Sweep 1: Hidden Dimension (fix: 9L, GQA 8Q/4KV)
# ==========================================================================
run_bench "D1: dim=448"  "dim"   "bench_dim448_elem.env"
run_bench "D2: dim=512"  "dim"   "bench_dim512_elem.env"
run_bench "D3: dim=768"  "dim"   "bench_dim768_elem.env"
run_bench "D4: dim=1024" "dim"   "bench_dim1024_elem.env"

# ==========================================================================
# Sweep 2: Number of Layers (fix: dim=512, GQA 8Q/4KV)
# D2 already ran above — reuse its results
# ==========================================================================
run_bench "L2: 10L"      "layer" "bench_10L_dim512_elem.env"
run_bench "L3: 11L"      "layer" "bench_11L_dim512_elem.env"

# ==========================================================================
# Sweep 3: GQA vs MHA (fix: dim=512, 9L)
# D2 already ran above — reuse its results
# ==========================================================================
run_bench "A2: MHA"       "attn"  "bench_mha_dim512_elem.env"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================
echo ""
echo "############################################################"
echo "  BENCHMARK SWEEP — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

runs = [
    ('D1', 'dim',   'bench_dim448',     'dim=448, 9L, GQA'),
    ('D2', 'dim',   'bench_dim512',     'dim=512, 9L, GQA'),
    ('D3', 'dim',   'bench_dim768',     'dim=768, 9L, GQA'),
    ('D4', 'dim',   'bench_dim1024',    'dim=1024, 9L, GQA'),
    ('L2', 'layer', 'bench_10L_dim512', 'dim=512, 10L, GQA'),
    ('L3', 'layer', 'bench_11L_dim512', 'dim=512, 11L, GQA'),
    ('A2', 'attn',  'bench_mha_dim512', 'dim=512, 9L, MHA'),
]

pg_dir = '$PG_DIR'

header = f'  {\"Run\":>3} | {\"Sweep\":<5} | {\"Config\":<20} | {\"Params\":>10} | {\"Steps\":>7} | {\"Pre-Q BPB\":>9} | {\"TTT BPB\":>8} | {\"GPTQ Size\":>10} | {\"GPTQ Time\":>9}'
sep    = f'  {\"-\"*3} | {\"-\"*5} | {\"-\"*20} | {\"-\"*10} | {\"-\"*7} | {\"-\"*9} | {\"-\"*8} | {\"-\"*10} | {\"-\"*9}'
print(header)
print(sep)

for label, sweep, run_id, desc in runs:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    na = '---'
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {sweep:<5} | {desc:<20} | {na:>10} | {na:>7} | {na:>9} | {na:>8} | {na:>10} | {na:>9}')
        continue
    text = open(log_file).read()

    # Params
    params = '?'
    m = re.search(r'model_params:(\d+)', text)
    if m: params = f'{int(m.group(1)):,}'

    # Steps
    steps_all = re.findall(r'^step:(\d+)/(\d+)', text, re.MULTILINE)
    steps = f'{steps_all[-1][0]}/{steps_all[-1][1]}' if steps_all else '?'

    # Pre-quant BPB (last training val)
    pre_bpb = '?'
    vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
    if vals: pre_bpb = vals[-1]

    # TTT BPB (int6 first, then int8)
    ttt_bpb = '?'
    m = re.search(r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: ttt_bpb = m.group(1)
    else:
        m = re.search(r'final_int8_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
        if m: ttt_bpb = m.group(1)

    # GPTQ size
    gptq_sz = '?'
    m = re.search(r'Serialized model int6\+brotli: (\d+) bytes', text)
    if m: gptq_sz = f'{int(m.group(1))/1e6:.2f}MB'

    # GPTQ time
    gptq_t = '?'
    m = re.search(r'gptq:done in ([\d.]+)s', text)
    if m: gptq_t = f'{float(m.group(1)):.0f}s'

    print(f'  {label:>3} | {sweep:<5} | {desc:<20} | {params:>10} | {steps:>7} | {pre_bpb:>9} | {ttt_bpb:>8} | {gptq_sz:>10} | {gptq_t:>9}')

print()
print('  D2 is shared baseline for all 3 sweeps (dim=512, 9L, GQA)')
print('  All runs: elementwise gated attn + GPTQ int7 (clip=63) + train data calib')
"

echo ""
echo "Done. Logs in parameter-golf/logs/"
