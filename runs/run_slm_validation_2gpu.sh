#!/bin/bash

# SLM Validation Runs — 4 focused experiments on 2×H100.
#
# Context: All previous "SLM" runs (Session 6) were INVALID — the RunPod had
# commit d7af1ec (Apr 23) which predates the SLM code push (Apr 26 7:18 PM).
# SLM_ENABLED env var was set but ignored by the code. These 4 runs are the
# FIRST actual SLM validation.
#
# Valid baselines (no SLM):
#   Run A  (SP8192 combo slim + TTT):  1.2411 BPB
#   Run H  (SP8192 combo slim no TTT): 1.2432 BPB
#   Run 6v2 (SP1024 GQA baseline):    1.2649 BPB
#
# Prerequisites:
#   - Repo cloned, on James-experiment, latest commit pulled
#   - VERIFY: grep -c "slm_enabled" parameter-golf/train_gpt.py  (must be >0)
#   - pip install -r requirements.txt, huggingface_hub, torch upgraded
#   - SP8192 and SP1024 datasets downloaded
#
# Usage (from repo root):
#   bash runs/run_slm_validation_2gpu.sh

set -uo pipefail

NGPUS=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# ---- Preflight: verify SLM code is present ----
SLM_HITS=$(grep -c "slm_enabled" "$PG_DIR/train_gpt.py" 2>/dev/null || echo 0)
if [ "$SLM_HITS" -eq 0 ]; then
    echo "FATAL: SLM code NOT found in train_gpt.py!"
    echo "  Expected: grep -c 'slm_enabled' parameter-golf/train_gpt.py > 0"
    echo "  Did you pull the latest James-experiment branch?"
    exit 1
fi
echo "Preflight OK: SLM code found ($SLM_HITS occurrences)"

# Track results
declare -a RUN_NAMES=()
declare -a RUN_IDS=()
declare -a RUN_STATUS=()

run_experiment() {
    local run_name="$1"
    local run_id="$2"
    shift 2

    echo ""
    echo "############################################################"
    echo "  $run_name"
    echo "  RUN_ID: $run_id | GPUs: $NGPUS"
    echo "############################################################"
    echo ""

    RUN_NAMES+=("$run_name")
    RUN_IDS+=("$run_id")

    export NGPUS=$NGPUS
    export RUN_ID="$run_id"
    for override in "$@"; do
        export "$override"
    done

    if bash "$REPO_ROOT/runs/parameter_golf_baseline.sh"; then
        RUN_STATUS+=("OK")
    else
        RUN_STATUS+=("FAIL")
        echo "WARNING: $run_name ($run_id) failed, continuing..."
    fi
}

echo "============================================"
echo "  SLM Validation — 2×H100"
echo "  $(date)"
echo "============================================"

cd "$REPO_ROOT"

# ==========================================================================
# S1: SP1024 GQA baseline + SLM k=0.6
# Smoke test: verify SLM works at all on the simplest config.
# Compare to Run 6v2 (1.2649 BPB, same config without SLM).
# ==========================================================================
source "$REPO_ROOT/runs/configs/explore_2gpu.env"
run_experiment \
    "S1: SP1024 GQA + SLM k=0.6 (smoke test)" \
    "slm_val_s1_sp1024_k60" \
    "SLM_ENABLED=1" \
    "SLM_RATIO=0.6"

# ==========================================================================
# S2: SP8192 combo slim + SLM k=0.6 + TTT
# Paper-recommended ratio on competition config.
# Compare to Run A (1.2411 BPB, same config without SLM).
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
run_experiment \
    "S2: SP8192 combo slim + SLM k=0.6" \
    "slm_val_s2_sp8192_k60" \
    "SLM_ENABLED=1" \
    "SLM_RATIO=0.6"

# ==========================================================================
# S3: SP8192 combo slim + SLM k=0.7 + TTT
# Middle ground between paper (0.6) and our hypothesis (0.8).
# Compare to Run A (1.2411 BPB).
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
run_experiment \
    "S3: SP8192 combo slim + SLM k=0.7" \
    "slm_val_s3_sp8192_k70" \
    "SLM_ENABLED=1" \
    "SLM_RATIO=0.7"

# ==========================================================================
# S4: SP8192 combo slim + SLM k=0.8 + TTT
# Our hypothesis: k=0.8 optimal for 17M scale.
# Compare to Run A (1.2411 BPB).
# ==========================================================================
source "$REPO_ROOT/runs/configs/sp8192_combo_slim.env"
run_experiment \
    "S4: SP8192 combo slim + SLM k=0.8" \
    "slm_val_s4_sp8192_k80" \
    "SLM_ENABLED=1" \
    "SLM_RATIO=0.8"

# ==========================================================================
# RESULTS SUMMARY
# ==========================================================================

echo ""
echo "############################################################"
echo "  SLM VALIDATION — RESULTS"
echo "  $(date)"
echo "############################################################"
echo ""

python3 -c "
import os, re

run_ids = [
    ('S1', 'slm_val_s1_sp1024_k60', 'SP1024 GQA + SLM k=0.6',   '1.2649'),
    ('S2', 'slm_val_s2_sp8192_k60', 'SP8192 combo + SLM k=0.6',  '1.2411'),
    ('S3', 'slm_val_s3_sp8192_k70', 'SP8192 combo + SLM k=0.7',  '1.2411'),
    ('S4', 'slm_val_s4_sp8192_k80', 'SP8192 combo + SLM k=0.8',  '1.2411'),
]

pg_dir = '$PG_DIR'

print(f'  {\"Run\":>3} | {\"Config\":<28} | {\"val_bpb\":>8} | {\"Baseline\":>8} | {\"Delta\":>8} | {\"Steps\":>6} | {\"Size\":>10}')
print(f'  {\"-\"*3} | {\"-\"*28} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*6} | {\"-\"*10}')

for label, run_id, desc, baseline in run_ids:
    log_file = os.path.join(pg_dir, 'logs', f'{run_id}.txt')
    if not os.path.isfile(log_file):
        print(f'  {label:>3} | {desc:<28} | {\"---\":>8} | {baseline:>8} | {\"---\":>8} | {\"---\":>6} | {\"---\":>10}')
        continue

    text = open(log_file).read()

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

    steps = re.findall(r'^step:(\d+)/\d+', text, re.MULTILINE)
    last_step = steps[-1] if steps else '?'

    comp = '?'
    m = re.search(r'Serialized model int8\+zlib: (\d+) bytes', text)
    if m:
        comp = f'{int(m.group(1))/1e6:.2f}MB'

    delta = '?'
    try:
        delta = f'{float(bpb) - float(baseline):+.4f}'
    except:
        pass

    print(f'  {label:>3} | {desc:<28} | {bpb:>8} | {baseline:>8} | {delta:>8} | {last_step:>6} | {comp:>10}')

print()
print('  Valid baselines (no SLM):')
print('    Run A  (SP8192 combo slim + TTT):  1.2411 BPB')
print('    Run H  (SP8192 combo slim no TTT): 1.2432 BPB')
print('    Run 6v2 (SP1024 GQA baseline):     1.2649 BPB')
print()
print('  PASS criteria:')
print('    S1: val_bpb < 1.2649 (Run 6v2)')
print('    S2-S4: val_bpb < 1.2411 (Run A)')
print('    Winner = lowest BPB among S2-S4 -> goes to 8xH100')
"

echo ""
echo "Done. Review results above — winner goes to 8xH100."
