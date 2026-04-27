#!/bin/bash

# Run Parameter Golf training 3 times with different seeds for submission.
# Uses the same config and logic as parameter_golf_baseline.sh.
#
# Usage:
#   source runs/configs/sp8192_combo_slim.env
#   NGPUS=8 bash runs/parameter_golf_8gpu_3seed_run.sh
#
# Seeds: 42, 1337, 2025 (PG submission standard)
# Each run produces a separate log: logs/${RUN_ID}_seed${SEED}.txt
# After all 3 runs, prints mean/std val_bpb for submission.json.

set -euo pipefail

SEEDS=(42 1337 2025)

# -----------------------------------------------------------------------------
# Configuration (inherit from env, same as parameter_golf_baseline.sh)

NGPUS="${NGPUS:-8}"
RUN_ID="${RUN_ID:-submission}"
TRAIN_SHARDS="${TRAIN_SHARDS:-}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
SP_VARIANT="sp${VOCAB_SIZE}"

# -----------------------------------------------------------------------------
# Locate parameter-golf repo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

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
# Download dataset (once, before the seed loop)

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
# Run 3 seeds

LOG_FILES=()

for SEED in "${SEEDS[@]}"; do
    SEED_RUN_ID="${RUN_ID}_seed${SEED}"

    echo ""
    echo "============================================"
    echo "  Parameter Golf — Seed $SEED (${SEED_RUN_ID})"
    echo "  GPUs: $NGPUS | Vocab: $VOCAB_SIZE"
    echo "============================================"
    echo ""

    export SEED
    export RUN_ID="$SEED_RUN_ID"
    export DATA_PATH="${DATA_PATH:-$DATA_DIR/}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-$TOKENIZER_FILE}"
    export VOCAB_SIZE

    torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

    LOG_FILE="logs/${SEED_RUN_ID}.txt"
    if [ -f "$LOG_FILE" ]; then
        LOG_FILES+=("$LOG_FILE")
        echo "==> Seed $SEED done. Log: $LOG_FILE"
    else
        echo "WARNING: Expected log not found at $LOG_FILE"
    fi

    echo ""
done

# Restore original RUN_ID for the summary
export RUN_ID="${RUN_ID%_seed*}"
RUN_ID="${RUN_ID:-submission}"

# -----------------------------------------------------------------------------
# Aggregate results across seeds

echo ""
echo "============================================"
echo "  3-SEED SUMMARY"
echo "============================================"
python3 -c "
import os, re, math

seeds = [42, 1337, 2025]
run_id = '${RUN_ID}'
results = []

for seed in seeds:
    log_file = f'logs/{run_id}_seed{seed}.txt'
    if not os.path.isfile(log_file):
        print(f'  WARNING: {log_file} not found, skipping seed {seed}')
        continue

    text = open(log_file).read()

    # Prefer TTT result, then int8+zlib roundtrip
    ttt_bpb = None
    ttt_loss = None
    m = re.search(r'final_int8_ttt_exact val_loss:([\d.]+) val_bpb:([\d.]+)', text)
    if m:
        ttt_loss = float(m.group(1))
        ttt_bpb = float(m.group(2))

    int8_bpb = None
    int8_loss = None
    m = re.search(r'final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)', text)
    if m:
        int8_loss = float(m.group(1))
        int8_bpb = float(m.group(2))

    final_bpb = ttt_bpb if ttt_bpb is not None else int8_bpb
    final_loss = ttt_loss if ttt_loss is not None else int8_loss

    # Steps
    steps = re.findall(r'^step:(\d+)/(\d+)', text, re.MULTILINE)
    last_step = steps[-1][0] if steps else '?'

    # Step avg (last value)
    step_avgs = re.findall(r'step_avg:([\d.]+)ms', text)
    step_avg = step_avgs[-1] if step_avgs else '?'

    # Artifact size
    comp_bytes = None
    m = re.search(r'Serialized model int8\+zlib: (\d+) bytes', text)
    if m:
        comp_bytes = int(m.group(1))

    # TTT time
    ttt_time = None
    m = re.search(r'TTT.*?total.*?([\d.]+)s', text, re.IGNORECASE)
    if m:
        ttt_time = float(m.group(1))

    results.append({
        'seed': seed,
        'val_bpb': final_bpb,
        'val_loss': final_loss,
        'steps': last_step,
        'step_avg': step_avg,
        'artifact_bytes': comp_bytes,
        'ttt_time': ttt_time,
    })

if not results:
    print('  No results found.')
    exit()

# Per-seed table
print()
print(f'  Run: {run_id}')
print()
print(f'  {\"Seed\":>6} | {\"Steps\":>7} | {\"step_avg\":>8} | {\"val_bpb\":>8} | {\"val_loss\":>8} | {\"Artifact\":>12}')
print(f'  {\"-\"*6} | {\"-\"*7} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*12}')
for r in results:
    art = f\"{r['artifact_bytes']:,}\" if r['artifact_bytes'] else '?'
    bpb = f\"{r['val_bpb']:.4f}\" if r['val_bpb'] else '?'
    loss = f\"{r['val_loss']:.4f}\" if r['val_loss'] else '?'
    print(f\"  {r['seed']:>6} | {r['steps']:>7} | {r['step_avg']:>6}ms | {bpb:>8} | {loss:>8} | {art:>12}\")

# Mean and std
bpbs = [r['val_bpb'] for r in results if r['val_bpb'] is not None]
losses = [r['val_loss'] for r in results if r['val_loss'] is not None]

if len(bpbs) >= 2:
    mean_bpb = sum(bpbs) / len(bpbs)
    std_bpb = math.sqrt(sum((x - mean_bpb)**2 for x in bpbs) / (len(bpbs) - 1))
    mean_loss = sum(losses) / len(losses)

    print(f'  {\"-\"*6} | {\"-\"*7} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*12}')
    print(f'  {\"Mean\":>6} |         |          | {mean_bpb:>8.4f} | {mean_loss:>8.4f} |')
    print(f'  {\"Std\":>6} |         |          | {std_bpb:>8.5f} |          |')
    print()
    print(f'  --- For submission.json ---')
    print(f'  \"val_bpb\": {mean_bpb:.4f},')
    print(f'  \"val_bpb_std\": {std_bpb:.5f},')
    print(f'  \"seeds\": {seeds},')
    print(f'  \"seed_results\": {{')
    for r in results:
        art = r['artifact_bytes'] if r['artifact_bytes'] else 0
        ttt = int(r['ttt_time'] * 1000) if r['ttt_time'] else 0
        print(f'    \"{r[\"seed\"]}\": {{\"val_bpb\": {r[\"val_bpb\"]:.4f}, \"val_loss\": {r[\"val_loss\"]:.4f}, \"artifact_bytes\": {art}, \"ttt_time_ms\": {ttt}}},')
    print(f'  }}')
    print()
    gap = mean_bpb - 1.2244
    print(f'  baseline:  1.2244')
    print(f'  gap:       {gap:+.4f}')
elif len(bpbs) == 1:
    print()
    print(f'  Only 1 seed completed — need all 3 for mean/std')
else:
    print()
    print(f'  No BPB results found')

print()
print(f'  Log files:')
for seed in seeds:
    lf = f'logs/{run_id}_seed{seed}.txt'
    exists = 'OK' if os.path.isfile(lf) else 'MISSING'
    print(f'    {lf} [{exists}]')
"
echo "============================================"
echo ""
echo "Done. All 3 seeds complete."
