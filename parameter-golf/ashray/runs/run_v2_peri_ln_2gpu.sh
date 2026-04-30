#!/bin/bash
#
# V2 — baseline + Peri-LN on 2×H100, 1 seed.
# Measures the delta from adding F.rms_norm on each sublayer output.
#
# Usage (from ashray/ root):
#   bash runs/run_v2_peri_ln_2gpu.sh
# Override seed:
#   SEED=0 bash runs/run_v2_peri_ln_2gpu.sh
#
# Compare against records/v1_baseline_seed${SEED}_2gpu/ for the delta.

set -uo pipefail

NGPUS="${NGPUS:-2}"
SEED="${SEED:-42}"
RUN_NAME="v2_peri_ln_seed${SEED}_${NGPUS}gpu"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASHRAY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$ASHRAY_ROOT/parameter-golf"
BASE_CONFIG="$ASHRAY_ROOT/runs/configs/v1_base.env"
OVERLAY_CONFIG="$ASHRAY_ROOT/runs/configs/v2_peri_ln.env"
TRAIN_SCRIPT="train_v2.py"
RECORDS_DIR="$PG_DIR/records/$RUN_NAME"

echo "============================================"
echo "  V2 — Peri-LN — ${NGPUS}×GPU, seed ${SEED}"
echo "  $(date)"
echo "============================================"

# --- Preflight ---
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found."
    exit 1
}
python3 -c "import brotli" 2>/dev/null || {
    echo "ERROR: brotli not found. pip install brotli"
    exit 1
}
python3 -c "import sentencepiece" 2>/dev/null || {
    echo "ERROR: sentencepiece not found. pip install sentencepiece"
    exit 1
}
echo "Preflight: FA3 + brotli + sentencepiece OK"

CASEOPS_SHARD="$PG_DIR/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin"
if [ ! -f "$CASEOPS_SHARD" ]; then
    echo "ERROR: CaseOps dataset not found. Run runs/configs/SETUP.md first."
    exit 1
fi
echo "Preflight: CaseOps dataset present"

# --- Load configs: base first, then overlay ---
source "$BASE_CONFIG"
source "$OVERLAY_CONFIG"
export RUN_ID="$RUN_NAME"
export SEED
export ARTIFACT_DIR="$RECORDS_DIR"

mkdir -p "$RECORDS_DIR"

echo ""
echo "Config summary:"
echo "  train_script:    $TRAIN_SCRIPT"
echo "  run_id:          $RUN_ID"
echo "  seed:            $SEED"
echo "  gpus:            $NGPUS"
echo "  records_dir:     $RECORDS_DIR"
echo "  peri_ln_enabled: $PERI_LN_ENABLED   <-- the delta vs v1 baseline"
echo ""

cd "$PG_DIR"

LOG_FILE="$RECORDS_DIR/train.log"
echo "==> Starting training. Logs: $LOG_FILE"
echo ""

torchrun --standalone --nproc_per_node="$NGPUS" "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Training exited with code $EXIT_CODE — see $LOG_FILE"
    exit "$EXIT_CODE"
fi

# --- Results summary ---
echo ""
echo "============================================"
echo "  RESULTS — $RUN_NAME"
echo "============================================"

python3 - <<PY
import os, re

log_file = "$LOG_FILE"
text = open(log_file).read()

def find(pat, default="?"):
    m = re.search(pat, text, re.MULTILINE)
    return m.group(1) if m else default

params       = find(r"model_params:(\d+)")
last_step    = (re.findall(r"^step:(\d+)/", text, re.MULTILINE) or ["?"])[-1]

pre_ema_bpb  = find(r"pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)")
quant_bpb    = find(r"^quantized val_loss:[\d.]+ val_bpb:([\d.]+)")
ttt_bpb      = find(r"quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)")

total_bytes  = find(r"Total submission size.*?: (\d+) bytes")
train_time   = find(r"stopping_early: wallclock_cap train_time:(\d+)")
vram         = find(r"peak memory allocated: (\d+) MiB")

print(f"  params:         {int(params):,}" if params != "?" else "  params:         ?")
print(f"  steps:          {last_step}")
print(f"  train_time_ms:  {train_time}")
print(f"  peak_vram_MiB:  {vram}")
print(f"  pre-TTT BPB:    {pre_ema_bpb}  (post-EMA, pre-quant)")
print(f"  quant BPB:      {quant_bpb}   (post-quant, pre-TTT)")
print(f"  post-TTT BPB:   {ttt_bpb}  <-- headline number")
print(f"  artifact bytes: {total_bytes}" + (f"  ({int(total_bytes)/1e6:.2f} MB)" if total_bytes != "?" else ""))
if total_bytes != "?":
    ok = "YES" if int(total_bytes) <= 16_000_000 else "NO"
    headroom = (16_000_000 - int(total_bytes)) / 1e6
    print(f"  under 16MB:     {ok}  ({headroom:+.2f} MB headroom)")

# Attempt to compute delta vs v1 baseline log if it exists.
v1_log = "records/v1_baseline_seed${SEED}_${NGPUS}gpu/train.log"
if os.path.exists(v1_log):
    v1_text = open(v1_log).read()
    v1_ttt = re.search(r"quantized_ttt val_loss:[\d.]+ val_bpb:([\d.]+)", v1_text)
    if v1_ttt and ttt_bpb != "?":
        delta = float(ttt_bpb) - float(v1_ttt.group(1))
        print()
        print(f"  v1 post-TTT:    {v1_ttt.group(1)}")
        print(f"  delta (v2-v1):  {delta:+.5f}  ({'BETTER' if delta < 0 else 'WORSE' if delta > 0 else 'TIE'})")
    else:
        print()
        print(f"  (couldn't extract v1 TTT BPB from {v1_log} for delta)")
else:
    print()
    print(f"  (no v1 baseline log at {v1_log} — run v1 first for delta comparison)")
PY

echo ""
echo "Done. Artifact at: $RECORDS_DIR/final_model.int6.ptz"
