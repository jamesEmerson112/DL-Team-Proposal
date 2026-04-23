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

DATA_DIR="./data/datasets/fineweb10B_sp1024"
TOKENIZER_FILE="./data/tokenizers/fineweb_1024_bpe.model"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "==> Downloading FineWeb sp1024 dataset..."
    DOWNLOAD_CMD="python3 data/cached_challenge_fineweb.py --variant sp1024"
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
echo "  Vocab: 1024 | Dataset: FineWeb sp1024"
echo "============================================"
echo ""

export RUN_ID
export DATA_PATH="$DATA_DIR/"
export TOKENIZER_PATH="$TOKENIZER_FILE"
export VOCAB_SIZE=1024

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

# Prefer int8+zlib compressed artifact (actual submission format)
compressed = glob.glob('*.int8.ptz') + glob.glob('*.ptz')
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
    label = 'int8+zlib'
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
step_avg = find(r'step_avg:([\d.]+)ms')
vram = find(r'peak memory allocated: (\d+) MiB')
gated = os.environ.get('GATED_ATTN', 'none')
activation = os.environ.get('ACTIVATION', 'relu2')
qk_gain = os.environ.get('QK_GAIN_INIT', '1.5')
ngpus = '${NGPUS}'

# Find last val_bpb and final int8 roundtrip
val_bpbs = re.findall(r'val_bpb:([\d.]+)', text)
last_raw_bpb = val_bpbs[-1] if val_bpbs else '?'
int8_bpb = find(r'final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)')

# Find last step
steps = re.findall(r'step:(\d+)/\d+', text)
last_step = steps[-1] if steps else '?'
total_steps = find(r'step:\d+/(\d+)')

# Find compressed size
comp_bytes = find(r'Serialized model int8\+zlib: (\d+) bytes')
comp_mb = f'{int(comp_bytes)/1e6:.2f}' if comp_bytes != '?' else '?'

budget_ok = 'YES' if comp_bytes != '?' and int(comp_bytes) + 120000 <= 16_000_000 else 'NO'

print(f'  run_id:       {run_id}')
print(f'  gpus:         {ngpus}')
print(f'  params:       {int(params):,}' if params != '?' else f'  params:       ?')
print(f'  gated_attn:   {gated}')
print(f'  activation:   {activation}')
print(f'  qk_gain_init: {qk_gain}')
print(f'  steps:        {last_step}/{total_steps}')
print(f'  step_avg:     {step_avg} ms')
print(f'  peak_vram:    {vram} MiB')
print(f'  val_bpb_raw:  {last_raw_bpb}')
print(f'  val_bpb_int8: {int8_bpb}')
print(f'  size_int8:    {comp_mb} MB')
print(f'  under_budget: {budget_ok}')
print(f'  baseline:     1.2244')
print(f'  gap:          +{float(int8_bpb) - 1.2244:.4f}' if int8_bpb != '?' else '  gap:          ?')
"
echo "============================================"
echo ""
echo "Done."
