#!/bin/bash

# Run the OpenAI Parameter Golf baseline (FineWeb sp1024, 1024 vocab)
# This is SEPARATE from nanochat — it uses openai/parameter-golf's train_gpt.py.
#
# Run as (from repo root):
#   bash runs/parameter_golf_baseline.sh
#
# For 8xH100 full run:
#   NGPUS=8 bash runs/parameter_golf_baseline.sh
#
# With fewer data shards (faster, for testing):
#   TRAIN_SHARDS=4 bash runs/parameter_golf_baseline.sh
#
# Competition: https://openai.com/index/parameter-golf/
# Repo:        https://github.com/openai/parameter-golf
# Deadline:    April 30, 2026
# Baseline:    1.2244 BPB (9 layers, 512 dims, 1024 vocab)

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via env vars)

NGPUS="${NGPUS:-1}"                          # GPUs to use (1 = smoke test, 8 = full run)
RUN_ID="${RUN_ID:-baseline_sp1024}"          # wandb / logging run name
TRAIN_SHARDS="${TRAIN_SHARDS:-}"            # empty = full dataset (80 shards)

# -----------------------------------------------------------------------------
# Locate parameter-golf repo (submodule in DL-Team-Proposal)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

# Clone if not present
if [ ! -d "$PG_DIR" ]; then
    echo "==> Cloning openai/parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$PG_DIR"
else
    echo "==> parameter-golf repo already present at $PG_DIR"
fi

cd "$PG_DIR"
echo "==> Working in $(pwd)"

# -----------------------------------------------------------------------------
# Install dependencies

if [ -f "requirements.txt" ]; then
    echo "==> Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
elif [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "==> Installing package..."
    pip install -e .
else
    echo "==> No requirements file found, installing common deps..."
    pip install torch sentencepiece
fi

# -----------------------------------------------------------------------------
# Download dataset (skip if already present)
# Supports SP1024 (default) or SP8192 via VOCAB_SIZE env var

VOCAB_SIZE="${VOCAB_SIZE:-1024}"
SP_VARIANT="sp${VOCAB_SIZE}"
DATA_DIR="./data/datasets/fineweb10B_${SP_VARIANT}"
TOKENIZER_FILE="./data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "==> Downloading FineWeb ${SP_VARIANT} dataset..."
    DOWNLOAD_CMD="python3 data/cached_challenge_fineweb.py --variant ${SP_VARIANT}"
    if [ -n "$TRAIN_SHARDS" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --train-shards $TRAIN_SHARDS"
    fi
    $DOWNLOAD_CMD
else
    echo "==> Dataset already present, skipping download."
fi

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "WARNING: Tokenizer not found at $TOKENIZER_FILE"
    echo "         The dataset download should have created it."
fi

# -----------------------------------------------------------------------------
# Train

echo ""
echo "============================================"
echo "  Parameter Golf — Training Baseline"
echo "  GPUs: $NGPUS | Run: $RUN_ID"
echo "  Vocab: $VOCAB_SIZE | Dataset: FineWeb ${SP_VARIANT}"
echo "============================================"
echo ""

export RUN_ID
export DATA_PATH="$DATA_DIR/"
export TOKENIZER_PATH="$TOKENIZER_FILE"
export VOCAB_SIZE

torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

# -----------------------------------------------------------------------------
# Remind to download log file for plotting

LOG_FILE="logs/${RUN_ID}.txt"
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "============================================"
    echo "  Log saved: $LOG_FILE"
    echo "  Download for plotting:"
    echo "    scp <pod>:$(pwd)/$LOG_FILE ."
    echo "    python tools/plot_curves.py $LOG_FILE --mode single"
    echo "============================================"
fi

# -----------------------------------------------------------------------------
# Results + artifact size check

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
echo ""

# Check artifact size (code + weights must fit in 16 MB = 16,000,000 bytes)
# PG submission uses int8+zlib compressed model — check that first, fall back to raw
python3 -c "
import os, glob

budget = 16_000_000  # 16 MB in bytes

# Prefer compressed artifact (int6+brotli or int8+zlib)
compressed = glob.glob('*.int6.ptb') + glob.glob('*.int8.ptz') + glob.glob('*.ptz')
raw_weights = glob.glob('*.pt') + glob.glob('*.pth') + glob.glob('*.bin') + glob.glob('model.*')
code_files = glob.glob('train_gpt*.py')

print('  Artifact components:')

# Show compressed size (submission-relevant)
comp_total = 0
for f in sorted(set(compressed)):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        comp_total += size
        print(f'    {f}: {size:,} bytes ({size/1e6:.2f} MB) [compressed]')

# Show raw size for reference
raw_total = 0
for f in sorted(set(raw_weights)):
    if os.path.isfile(f) and f not in compressed:
        size = os.path.getsize(f)
        raw_total += size
        print(f'    {f}: {size:,} bytes ({size/1e6:.2f} MB) [raw]')

code_total = 0
for f in sorted(set(code_files)):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        code_total += size
        print(f'    {f}: {size:,} bytes ({size/1e6:.2f} MB) [code]')

# Use compressed if available, otherwise raw
if comp_total > 0:
    submission = comp_total + code_total
    label = 'int6+brotli' if any('.int6.ptb' in f for f in compressed) else 'int8+zlib'
else:
    submission = raw_total + code_total
    label = 'raw (no compressed artifact found)'

print(f'')
print(f'  Submission size ({label}): {submission:,} bytes ({submission/1e6:.2f} MB)')
print(f'  Budget:                    {budget:,} bytes (16.00 MB)')
if submission > 0:
    headroom = (budget - submission) / 1e6
    status = 'YES' if submission <= budget else 'NO'
    print(f'  Under budget:              {status} ({headroom:+.2f} MB headroom)')
else:
    print(f'  (no weight files found -- check training output)')
print()
print(f'  Baseline to beat:    1.2244 BPB')
"

# -----------------------------------------------------------------------------
# Compact summary — paste this block to log the run

echo ""
echo "============================================"
echo "  COPY-PASTE SUMMARY"
echo "============================================"
python3 -c "
import os, re

log_file = 'logs/${RUN_ID}.txt'
if not os.path.isfile(log_file):
    print('  (log file not found)')
    exit()

lines = open(log_file).readlines()
text = ''.join(lines)

# Extract key fields from log
def find(pattern, default='?'):
    m = re.search(pattern, text)
    return m.group(1) if m else default

run_id = '${RUN_ID}'
params = find(r'model_params:(\d+)')
heads = find(r'num_heads:(\d+)')
kv_heads = find(r'num_kv_heads:(\d+)')
# Use last step_avg (first one is 0.01ms from step 0)
step_avgs = re.findall(r'step_avg:([\d.]+)ms', text)
step_avg = step_avgs[-1] if step_avgs else '?'
vram = find(r'peak memory allocated: (\d+) MiB')
gated = os.environ.get('GATED_ATTN', 'none')
activation = os.environ.get('ACTIVATION', 'relu2')
qk_gain = os.environ.get('QK_GAIN_INIT', '1.5')
ttt_mode = os.environ.get('TTT_MODE', 'none')
quant_mode = os.environ.get('QUANT_MODE', 'int8_zlib')
slm_enabled = os.environ.get('SLM_ENABLED', '0')
slm_ratio = os.environ.get('SLM_RATIO', '0.6')
use_gptq = os.environ.get('USE_GPTQ', '0')
vocab_size = os.environ.get('VOCAB_SIZE', '1024')
ngpus = '${NGPUS}'

# Find last val_loss/val_bpb from TRAINING (exclude roundtrip lines)
# Training lines look like: step:3500/20000 val_loss:2.1338 val_bpb:1.2638
# Roundtrip lines look like: final_int8_zlib_roundtrip val_loss:...
train_vals = re.findall(r'^step:\d+/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)', text, re.MULTILINE)
last_raw_loss = train_vals[-1][0] if train_vals else '?'
last_raw_bpb = train_vals[-1][1] if train_vals else '?'
# Roundtrip results — try int6 first, then int8
int8_loss = find(r'final_int6_brotli_roundtrip_exact val_loss:([\d.]+)')
if int8_loss == '?':
    int8_loss = find(r'final_int8_zlib_roundtrip_exact val_loss:([\d.]+)')
int8_bpb = find(r'final_int6_brotli_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)')
if int8_bpb == '?':
    int8_bpb = find(r'final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)')

# GPTQ timing (if enabled)
gptq_time = find(r'gptq:done in ([\d.]+)s')

# TTT results (if enabled) — try int6 first, then int8
ttt_loss = find(r'final_int6_ttt_exact val_loss:([\d.]+)')
if ttt_loss == '?':
    ttt_loss = find(r'final_int8_ttt_exact val_loss:([\d.]+)')
ttt_bpb = find(r'final_int6_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)')
if ttt_bpb == '?':
    ttt_bpb = find(r'final_int8_ttt_exact val_loss:[\d.]+ val_bpb:([\d.]+)')

# Find last step (use ^step: to exclude warmup_step: lines)
steps = re.findall(r'^step:(\d+)/(\d+)', text, re.MULTILINE)
last_step = steps[-1][0] if steps else '?'
total_steps = steps[-1][1] if steps else '?'

# Find compressed size — try int6+brotli first, then int8+zlib
comp_bytes = find(r'Serialized model int6\+brotli: (\d+) bytes')
if comp_bytes == '?':
    comp_bytes = find(r'Serialized model int8\+zlib: (\d+) bytes')
comp_mb = f'{int(comp_bytes)/1e6:.2f}' if comp_bytes != '?' else '?'

budget_ok = 'YES' if comp_bytes != '?' and int(comp_bytes) + 120000 <= 16_000_000 else 'NO'

print(f'  run_id:       {run_id}')
print(f'  gpus:         {ngpus}')
print(f'  params:       {int(params):,}' if params != '?' else f'  params:       ?')
print(f'  gated_attn:   {gated}')
print(f'  activation:   {activation}')
print(f'  qk_gain_init: {qk_gain}')
print(f'  ttt_mode:     {ttt_mode}')
print(f'  quant_mode:   {quant_mode}')
print(f'  slm_enabled:  {slm_enabled}')
print(f'  slm_ratio:    {slm_ratio}')
print(f'  use_gptq:     {use_gptq}')
print(f'  vocab_size:   {vocab_size}')
print(f'  steps:        {last_step}/{total_steps}')
print(f'  step_avg:     {step_avg} ms')
print(f'  peak_vram:    {vram} MiB')
print(f'  val_loss_raw: {last_raw_loss}')
print(f'  val_bpb_raw:  {last_raw_bpb}')
print(f'  val_loss_int8:{int8_loss}')
print(f'  val_bpb_int8: {int8_bpb}')
print(f'  size_int8:    {comp_mb} MB')
print(f'  under_budget: {budget_ok}')
if gptq_time != '?':
    print(f'  gptq_time:    {gptq_time}s')
if ttt_bpb != '?':
    print(f'  val_loss_ttt: {ttt_loss}')
    print(f'  val_bpb_ttt:  {ttt_bpb}')
print(f'  baseline:     1.2244')
# Use TTT BPB as final result if available, otherwise int8
final_bpb = ttt_bpb if ttt_bpb != '?' else int8_bpb
print(f'  gap:          +{float(final_bpb) - 1.2244:.4f}' if final_bpb != '?' else '  gap:          ?')
"
echo "============================================"
echo ""
echo "Done."
