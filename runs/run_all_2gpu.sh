#!/bin/bash

# Run all 2xH100 experiments sequentially — fire and forget.
# Includes SP8192 combo slim retry + SLM phases 1-3.
#
# Prerequisites (already done before running this script):
#   - Repo cloned and on learning-curve-chart branch
#   - pip install -r requirements.txt, huggingface_hub, torch upgraded
#   - SP8192 and SP1024 datasets downloaded
#
# Usage (from repo root):
#   bash runs/run_all_2gpu.sh
#
# Each run sources its own config, so env vars are reset between runs.
# Results summary printed at the end.

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# Track which runs succeeded/failed
declare -a RUN_NAMES=()
declare -a RUN_IDS=()
declare -a RUN_STATUS=()

run_experiment() {
    local run_name="$1"
    local run_id="$2"
    shift 2
    # remaining args are env var overrides: KEY=VALUE

    echo ""
    echo "############################################################"
    echo "  $run_name"
    echo "  RUN_ID: $run_id | GPUs: $NGPUS"
    echo "############################################################"
    echo ""

    RUN_NAMES+=("$run_name")
    RUN_IDS+=("$run_id")

    # Apply env var overrides
    export NGPUS=$NGPUS
    export RUN_ID="$run_id"
    for override in "$@"; do
        export "$override"
    done

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        RUN_STATUS+=("OK")
    else
        RUN_STATUS+=("FAIL")
        echo "WARNING: $run_name ($run_id) failed, continuing to next run..."
    fi
}

echo "============================================"
echo "  2xH100 Experiment Suite"
echo "  $(date)"
echo "  Assumes deps installed + datasets downloaded"
echo "============================================"

cd "$REPO_ROOT"

# ==========================================================================
# RUN A: SP8192 combo slim retry (2xH100)
# Run 12 was accidental vanilla baseline — config source failed (stale env).
# This is the proper retry with all techniques.
# Compare against Run 11 (8xH100, 1.2077 BPB) to see 2-vs-8 GPU scaling.
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
run_experiment \
    "Run A: SP8192 combo slim retry" \
    "sp8192_combo_slim_2gpu_v2"

# ==========================================================================
# RUN B: SLM Smoke Test (Phase 1)
# Keeps top 60% tokens by loss, skips easy ones.
# SP1024 baseline to compare against Run 6v2 (1.2649 BPB).
# PASS: val_bpb improves AND step_avg doesn't regress >5%.
# ==========================================================================
source "$REPO_ROOT/runs/configs/slm_test.env"
run_experiment \
    "Run B: SLM smoke test (k=0.6)" \
    "slm_test_k60_2gpu"

# ==========================================================================
# RUN C: SLM Ratio Sweep (Phase 2)
# Test k=0.5, 0.7, 0.8 to find the sweet spot for 17M scale.
# PASS: at least one ratio beats baseline by >0.002 BPB.
# ==========================================================================

source "$REPO_ROOT/runs/configs/slm_sweep_50.env"
run_experiment \
    "Run C1: SLM sweep k=0.5" \
    "slm_sweep_k50_2gpu"

source "$REPO_ROOT/runs/configs/slm_sweep_70.env"
run_experiment \
    "Run C2: SLM sweep k=0.7" \
    "slm_sweep_k70_2gpu"

source "$REPO_ROOT/runs/configs/slm_sweep_80.env"
run_experiment \
    "Run C3: SLM sweep k=0.8" \
    "slm_sweep_k80_2gpu"

# ==========================================================================
# RUN D: SLM + SP8192 combo slim (Phase 3)
# Stack SLM with full competition config.
# TODO: replace SLM_RATIO with the winner from Phase 2 runs above.
# PASS: val_bpb improves over Run 11 (1.2077) AND fits 16 MB.
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
run_experiment \
    "Run D: SP8192 combo slim + SLM" \
    "sp8192_combo_slim_slm_2gpu" \
    "SLM_ENABLED=1" \
    "SLM_RATIO=0.6"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================

echo ""
echo "############################################################"
echo "  ALL RUNS COMPLETE — RESULTS SUMMARY"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re, glob

run_ids = [
    ('A',  'sp8192_combo_slim_2gpu_v2',   'SP8192 combo slim retry'),
    ('B',  'slm_test_k60_2gpu',           'SLM smoke test k=0.6'),
    ('C1', 'slm_sweep_k50_2gpu',          'SLM sweep k=0.5'),
    ('C2', 'slm_sweep_k70_2gpu',          'SLM sweep k=0.7'),
    ('C3', 'slm_sweep_k80_2gpu',          'SLM sweep k=0.8'),
    ('D',  'sp8192_combo_slim_slm_2gpu',  'SP8192 combo slim + SLM'),
]

pg_dir = '$PG_DIR'

print(f'  {\"Run\":>3} | {\"Config\":<30} | {\"val_bpb\":>8} | {\"Steps\":>7} | {\"step_avg\":>8} | {\"Size\":>10} | Status')
print(f'  {\"-\"*3} | {\"-\"*30} | {\"-\"*8} | {\"-\"*7} | {\"-\"*8} | {\"-\"*10} | ------')

for label, run_id, desc in run_ids:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<30} | {\"---\":>8} | {\"---\":>7} | {\"---\":>8} | {\"---\":>10} | NO LOG')
        continue

    text = open(log_file).read()

    # Prefer TTT BPB, then int8+zlib, then raw training
    bpb = '?'
    m = re.search(r'final_int8_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m:
        bpb = m.group(1)
    else:
        m = re.search(r'final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)', text)
        if m:
            bpb = m.group(1)
        else:
            vals = re.findall(r'^step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)', text, re.MULTILINE)
            if vals:
                bpb = vals[-1]

    # Steps
    steps = re.findall(r'^step:(\d+)/\d+', text, re.MULTILINE)
    last_step = steps[-1] if steps else '?'

    # Step avg
    step_avgs = re.findall(r'step_avg:([\d.]+)ms', text)
    step_avg = step_avgs[-1] + 'ms' if step_avgs else '?'

    # Compressed size
    comp = '?'
    m = re.search(r'Serialized model int8\+zlib: (\d+) bytes', text)
    if m:
        comp = f'{int(m.group(1))/1e6:.2f}MB'

    print(f'  {label:>3} | {desc:<30} | {bpb:>8} | {last_step:>7} | {step_avg:>8} | {comp:>10} | OK')

print()
print('  Baselines for comparison:')
print('    Run 6v2 (SP1024 GQA baseline):     1.2649 BPB')
print('    Run 11  (SP8192 combo slim + TTT):  1.2077 BPB')
print('    PG baseline:                        1.2244 BPB')
"

echo ""
echo "############################################################"
echo ""
echo "Done. All experiments finished."
