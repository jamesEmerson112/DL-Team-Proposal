#!/bin/bash
#
# V1 baseline — rank 4 unmodified — on 2×H100, 1 seed.
# Our anchor for all subsequent delta experiments.
#
# Usage (from ashray/ root):
#   bash runs/run_v1_baseline_2gpu.sh
# Override seed:
#   SEED=0 bash runs/run_v1_baseline_2gpu.sh
# Override GPU count (smoke test on 1 GPU):
#   NGPUS=1 bash runs/run_v1_baseline_2gpu.sh
#
# Rank 4 source: PR #1769, 1.0645 BPB 5-seed mean on 8×H100.
# Expected on 2×H100: higher BPB (fewer training steps under the same 600s wallclock).

set -uo pipefail

NGPUS="${NGPUS:-2}"
SEED="${SEED:-42}"
RUN_NAME="v1_baseline_seed${SEED}_${NGPUS}gpu"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASHRAY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$ASHRAY_ROOT/parameter-golf"
BASE_CONFIG="$ASHRAY_ROOT/runs/configs/v1_base.env"
TRAIN_SCRIPT="train_v1.py"
RECORDS_DIR="$PG_DIR/records/$RUN_NAME"

echo "============================================"
echo "  V1 Baseline — ${NGPUS}×GPU, seed ${SEED}"
echo "  $(date)"
echo "============================================"

# --- Preflight ---
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "ERROR: flash_attn_3 not found. Install per runs/configs/README or the Runpod PG template."
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

# --- Dataset check (vanilla SP8192 must exist before training) ---
SHARD_PATH="$PG_DIR/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
TOKENIZER_PATH="$PG_DIR/data/tokenizers/fineweb_8192_bpe.model"
if [ ! -f "$SHARD_PATH" ] || [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Vanilla SP8192 dataset not found."
    echo "       Expected shard:     $SHARD_PATH"
    echo "       Expected tokenizer: $TOKENIZER_PATH"
    echo "       Run the setup steps in runs/configs/SETUP.md first:"
    echo "         rm -f parameter-golf/data/manifest.json"
    echo "         MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\"
    echo "           python3 parameter-golf/data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128"
    exit 1
fi
echo "Preflight: vanilla SP8192 dataset present"

# --- Load base config + per-run overrides ---
source "$BASE_CONFIG"
export RUN_ID="$RUN_NAME"
export SEED
export ARTIFACT_DIR="$RECORDS_DIR"

mkdir -p "$RECORDS_DIR"

echo ""
echo "Config summary:"
echo "  train_script: $TRAIN_SCRIPT"
echo "  run_id:       $RUN_ID"
echo "  seed:         $SEED"
echo "  gpus:         $NGPUS"
echo "  records_dir:  $RECORDS_DIR"
echo "  data_dir:     $DATA_DIR"
echo "  caseops:      $CASEOPS_ENABLED"
echo ""

# --- Train ---
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
steps_stop   = find(r"stopping_early: wallclock_cap .*?step:(\d+)")
steps_final  = find(r"^step:(\d+)/\d+ train_loss", )
last_step    = steps_stop if steps_stop != "?" else (re.findall(r"^step:(\d+)/", text, re.MULTILINE) or ["?"])[-1]

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
    budget_ok = "YES" if int(total_bytes) <= 16_000_000 else "NO"
    headroom = (16_000_000 - int(total_bytes)) / 1e6
    print(f"  under 16MB:     {budget_ok}  ({headroom:+.2f} MB headroom)")
print()
print(f"  Rank 4 published (5-seed 8×H100): 1.0645")
print(f"  Rank 3 published (3-seed 8×H100): 1.0634  (rank 4 + MIN_LR + 3 others)")
print(f"  Our v1 baseline (1-seed 2×H100):  post-TTT {ttt_bpb}  (rank 4 + MIN_LR)")
PY

echo ""
echo "Done. Results logged to $LOG_FILE"
echo "Artifact at: $RECORDS_DIR/final_model.int6.ptz (if training completed)"
