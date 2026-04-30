#!/bin/bash
#
# V4 — baseline + Universal-Transformer timestep embed on depth recurrence.
# Measures the delta vs v1 from giving looped layers a per-iteration signal.
#
# Usage (from ashray/ root):
#   bash runs/run_v4_ts_embed_2gpu.sh
# Override seed:
#   SEED=0 bash runs/run_v4_ts_embed_2gpu.sh

set -uo pipefail

NGPUS="${NGPUS:-2}"
SEED="${SEED:-42}"
RUN_NAME="v4_ts_embed_seed${SEED}_${NGPUS}gpu"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASHRAY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$ASHRAY_ROOT/parameter-golf"
BASE_CONFIG="$ASHRAY_ROOT/runs/configs/v1_base.env"
OVERLAY_CONFIG="$ASHRAY_ROOT/runs/configs/v4_ts_embed.env"
TRAIN_SCRIPT="train_v4.py"
RECORDS_DIR="$PG_DIR/records/$RUN_NAME"

echo "============================================"
echo "  V4 — TS-embed — ${NGPUS}×GPU, seed ${SEED}"
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

SHARD_PATH="$PG_DIR/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
TOKENIZER_PATH="$PG_DIR/data/tokenizers/fineweb_8192_bpe.model"
if [ ! -f "$SHARD_PATH" ] || [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Vanilla SP8192 dataset not found. Run runs/configs/SETUP.md first."
    exit 1
fi
echo "Preflight: vanilla SP8192 dataset present"

# --- Load configs: base first, then overlay ---
source "$BASE_CONFIG"
source "$OVERLAY_CONFIG"
export RUN_ID="$RUN_NAME"
export SEED
export ARTIFACT_DIR="$RECORDS_DIR"

mkdir -p "$RECORDS_DIR"

echo ""
echo "Config summary:"
echo "  train_script:     $TRAIN_SCRIPT"
echo "  run_id:           $RUN_ID"
echo "  seed:             $SEED"
echo "  gpus:             $NGPUS"
echo "  records_dir:      $RECORDS_DIR"
echo "  TS_EMBED_ENABLED: $TS_EMBED_ENABLED   <-- delta vs v1"
echo "  NUM_LOOPS:        $NUM_LOOPS"
echo "  LOOP_START:       $LOOP_START"
echo "  LOOP_END:         $LOOP_END"
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

pre_ema_bpb  = find(r"diagnostic pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)")
quant_bpb    = find(r"^diagnostic quantized val_loss:[\d.]+ val_bpb:([\d.]+)")
ttt_bpb      = find(r"quantized_ttt_phased val_loss:[\d.]+ val_bpb:([\d.]+)")

total_bytes  = find(r"Total submission size.*?: (\d+) bytes")
train_time   = find(r"stopping_early: wallclock_cap train_time:(\d+)")
vram         = find(r"peak memory allocated: (\d+) MiB")

print(f"  params:         {int(params):,}" if params != "?" else "  params:         ?")
print(f"  steps:          {last_step}")
print(f"  train_time_ms:  {train_time}")
print(f"  peak_vram_MiB:  {vram}")
print(f"  pre-TTT BPB:    {pre_ema_bpb}  (post-EMA, pre-quant)")
print(f"  quant BPB:      {quant_bpb}   (post-quant, pre-TTT)")
print(f"  post-TTT BPB:   {ttt_bpb}  <-- headline")
print(f"  artifact bytes: {total_bytes}" + (f"  ({int(total_bytes)/1e6:.2f} MB)" if total_bytes != "?" else ""))
if total_bytes != "?":
    ok = "YES" if int(total_bytes) <= 16_000_000 else "NO"
    headroom = (16_000_000 - int(total_bytes)) / 1e6
    print(f"  under 16MB:     {ok}  ({headroom:+.2f} MB headroom)")

v1_log = "records/v1_baseline_seed${SEED}_${NGPUS}gpu/train.log"
if os.path.exists(v1_log):
    v1_text = open(v1_log).read()
    v1_ttt = re.search(r"quantized_ttt_phased val_loss:[\d.]+ val_bpb:([\d.]+)", v1_text)
    v1_pre = re.search(r"diagnostic pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)", v1_text)
    if v1_pre and pre_ema_bpb != "?":
        d = float(pre_ema_bpb) - float(v1_pre.group(1))
        print()
        print(f"  v1 pre-TTT:     {v1_pre.group(1)}")
        print(f"  Δ pre-TTT:      {d:+.5f}  ({'BETTER' if d < 0 else 'WORSE' if d > 0 else 'TIE'})")
    if v1_ttt and ttt_bpb != "?":
        d = float(ttt_bpb) - float(v1_ttt.group(1))
        print(f"  v1 post-TTT:    {v1_ttt.group(1)}")
        print(f"  Δ post-TTT:     {d:+.5f}  ({'BETTER' if d < 0 else 'WORSE' if d > 0 else 'TIE'})")
PY

echo ""
echo "Done. Artifact at: $RECORDS_DIR/final_model.int6.ptz"
