#!/bin/bash
# =============================================================================
# C6 Submission + Ablation Runs on 8xH100 (6 runs total)
#
# Part A: 3-seed C6 submission (S1-S3)
# Part B: 3 ablation runs (A1-A3)
#
# Usage (from parameter-golf/ directory):
#   bash ../runs/run_v2_c6_8gpu.sh
#
# Estimated time: ~80 min (6 runs x ~13 min each)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

# =============================================================================
# PART A: 3-SEED C6 SUBMISSION
# =============================================================================

SEEDS=(42 1337 2025)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================"
    echo "  C6 Submission — Seed $SEED"
    echo "============================================"
    echo ""

    source "$REPO_ROOT/runs/configs/v2_base.env"
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    export SEED=$SEED
    export RUN_ID=v2_c6_seed${SEED}

    echo "VERIFY: GATED_ATTN=$GATED_ATTN EMBED_BITS=$EMBED_BITS EMBED_CLIP_SIGMAS=$EMBED_CLIP_SIGMAS SEED=$SEED RUN_ID=$RUN_ID"
    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py

    echo "==> C6 seed $SEED done. Log: logs/${RUN_ID}.txt"
done

# =============================================================================
# PART B: 3 ABLATION RUNS (seed 42)
# =============================================================================

# --- A1: F1 control (rank 1 defaults, no additions) ---
echo ""
echo "============================================"
echo "  Ablation A1: F1 control (no additions)"
echo "============================================"
echo ""
source "$REPO_ROOT/runs/configs/v2_base.env"
export SEED=42
export RUN_ID=v2_f1_8gpu
echo "VERIFY: GATED_ATTN=$GATED_ATTN VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA RUN_ID=$RUN_ID"
torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
echo "==> A1 done. Log: logs/${RUN_ID}.txt"

# --- A2: F7 (PR+RF, no gate) ---
echo ""
echo "============================================"
echo "  Ablation A2: F7 (ResFormer alpha=0.5)"
echo "============================================"
echo ""
source "$REPO_ROOT/runs/configs/v2_base.env"
export VALUE_RESIDUAL_ALPHA=0.5
export SEED=42
export RUN_ID=v2_f7_8gpu
echo "VERIFY: GATED_ATTN=$GATED_ATTN VALUE_RESIDUAL_ALPHA=$VALUE_RESIDUAL_ALPHA RUN_ID=$RUN_ID"
torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
echo "==> A2 done. Log: logs/${RUN_ID}.txt"

# --- A3: F2 (headwise gate, default compression — over budget) ---
echo ""
echo "============================================"
echo "  Ablation A3: F2 (headwise, no compression tune)"
echo "============================================"
echo ""
source "$REPO_ROOT/runs/configs/v2_base.env"
export GATED_ATTN=headwise
export SEED=42
export RUN_ID=v2_c6_nogptqtune
echo "VERIFY: GATED_ATTN=$GATED_ATTN EMBED_BITS=$EMBED_BITS EMBED_CLIP_SIGMAS=$EMBED_CLIP_SIGMAS RUN_ID=$RUN_ID"
torchrun --standalone --nproc_per_node=8 train_gpt_v2.py
echo "==> A3 done. Log: logs/${RUN_ID}.txt"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  ALL 6 RUNS COMPLETE — SUMMARY"
echo "============================================"

python3 -c "
import os, re, math

# --- Part A: 3-seed C6 ---
seeds = [42, 1337, 2025]
run_prefix = 'v2_c6'
results = []

for seed in seeds:
    log_file = f'logs/{run_prefix}_seed{seed}.txt'
    if not os.path.isfile(log_file):
        print(f'  WARNING: {log_file} not found, skipping seed {seed}')
        continue
    text = open(log_file).read()

    ttt_bpb = ttt_loss = None
    m = re.search(r'quantized_ttt val_loss:([\d.]+) val_bpb:([\d.]+)', text)
    if m:
        ttt_loss = float(m.group(1))
        ttt_bpb = float(m.group(2))

    quant_bytes = None
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m:
        quant_bytes = int(m.group(1))

    total_bytes = None
    m = re.search(r'Total submission size quantized\+\w+: (\d+) bytes', text)
    if m:
        total_bytes = int(m.group(1))

    steps = re.findall(r'(\d+)/\d+ val_loss:', text)
    last_step = steps[-1] if steps else '?'

    results.append({
        'seed': seed,
        'val_bpb': ttt_bpb,
        'val_loss': ttt_loss,
        'steps': last_step,
        'quant_bytes': quant_bytes,
        'total_bytes': total_bytes,
    })

print()
print(f'  === C6 3-SEED RESULTS (8xH100) ===')
print()
print(f'  {\"Seed\":>6} | {\"Steps\":>7} | {\"TTT BPB\":>9} | {\"TTT Loss\":>9} | {\"Weights\":>12} | Budget?')
print(f'  {\"-\"*6} | {\"-\"*7} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} | -------')
for r in results:
    bpb = f\"{r['val_bpb']:.4f}\" if r['val_bpb'] else '?'
    loss = f\"{r['val_loss']:.6f}\" if r['val_loss'] else '?'
    qb = f\"{r['quant_bytes']:,}\" if r['quant_bytes'] else '?'
    budget = 'Yes' if r['quant_bytes'] and r['quant_bytes'] < 16_000_000 else 'No/?'
    print(f\"  {r['seed']:>6} | {r['steps']:>7} | {bpb:>9} | {loss:>9} | {qb:>12} | {budget}\")

bpbs = [r['val_bpb'] for r in results if r['val_bpb'] is not None]
losses = [r['val_loss'] for r in results if r['val_loss'] is not None]

if len(bpbs) >= 2:
    mean_bpb = sum(bpbs) / len(bpbs)
    std_bpb = math.sqrt(sum((x - mean_bpb)**2 for x in bpbs) / (len(bpbs) - 1))
    mean_loss = sum(losses) / len(losses)

    print(f'  {\"-\"*6} | {\"-\"*7} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} |')
    print(f'  {\"Mean\":>6} |         | {mean_bpb:>9.4f} | {mean_loss:>9.6f} |              |')
    print(f'  {\"Std\":>6} |         | {std_bpb:>9.5f} |           |              |')
    print()
    print(f'  --- For submission.json ---')
    print(f'  \"val_bpb\": {mean_bpb:.4f},')
    print(f'  \"val_bpb_std\": {std_bpb:.5f},')
    print(f'  \"seeds\": {seeds},')
    print(f'  \"seed_results\": {{')
    for r in results:
        qb = r['quant_bytes'] if r['quant_bytes'] else 0
        print(f'    \"{r[\"seed\"]}\": {{\"val_bpb\": {r[\"val_bpb\"]:.4f}, \"val_loss\": {r[\"val_loss\"]:.6f}, \"quant_bytes\": {qb}}},')
    print(f'  }}')
    print()
    print(f'  PG baseline:  1.2244')
    print(f'  Our 2xH100:   1.1622 (C6)')
    print(f'  8xH100 mean:  {mean_bpb:.4f}')
    print(f'  Gap vs PG:    {mean_bpb - 1.2244:+.4f}')

# --- Part B: Ablation ---
print()
print(f'  === ABLATION RESULTS (8xH100, seed 42) ===')
print()

ablations = [
    ('A1 (F1 control)',      'v2_f1_8gpu'),
    ('A2 (F7 PR+RF)',        'v2_f7_8gpu'),
    ('A3 (F2 headwise)',     'v2_c6_nogptqtune'),
    ('S1 (C6 submission)',   'v2_c6_seed42'),
]

print(f'  {\"Run\":>22} | {\"TTT BPB\":>9} | {\"TTT Loss\":>9} | {\"Weights\":>12} | Budget?')
print(f'  {\"-\"*22} | {\"-\"*9} | {\"-\"*9} | {\"-\"*12} | -------')

for label, run_id in ablations:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'  {label:>22} | {\"MISSING\":>9} |           |              |')
        continue
    text = open(log_file).read()

    ttt_bpb = ttt_loss = None
    m = re.search(r'quantized_ttt val_loss:([\d.]+) val_bpb:([\d.]+)', text)
    if m:
        ttt_loss = float(m.group(1))
        ttt_bpb = float(m.group(2))

    quant_bytes = None
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m:
        quant_bytes = int(m.group(1))

    bpb = f'{ttt_bpb:.4f}' if ttt_bpb else '?'
    loss = f'{ttt_loss:.6f}' if ttt_loss else '?'
    qb = f'{quant_bytes:,}' if quant_bytes else '?'
    budget = 'Yes' if quant_bytes and quant_bytes < 16_000_000 else 'No/?'
    print(f'  {label:>22} | {bpb:>9} | {loss:>9} | {qb:>12} | {budget}')

print()
print(f'  Technique contributions (lower BPB = better):')
print(f'    Headwise gate effect:     A1 vs A3 (F1 vs F2)')
print(f'    ResFormer effect:         A1 vs A2 (F1 vs F7)')
print(f'    Compression tuning cost:  A3 vs S1 (F2 vs C6)')
print()
print(f'  2xH100 reference: F1=1.1641, F2=1.1636, F7=1.1636, C6=1.1622')
"

echo ""
echo "============================================"
echo "  Done. All 6 runs complete."
echo "============================================"
