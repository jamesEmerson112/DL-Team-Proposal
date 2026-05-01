#!/bin/bash
# =============================================================================
# X1 Submission — 3-Seed on 8xH100
#
# Full Stack: Small Batch + EMA=0.990 + Headwise Gate + PreQuantTTT
# 2xH100 result: 1.0591 TTT BPB
# Projected 8xH100: ~0.98-1.00 BPB
#
# Usage: bash runs/run_v2_x1_8gpu.sh
# Estimated time: ~45 min (3 seeds x ~15 min each)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

SEEDS=(42 1337 2025)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================"
    echo "  X1 Full Stack — Seed $SEED"
    echo "============================================"

    # Load configs: base → C6 overrides → full stack
    source "$REPO_ROOT/runs/configs/v2_base.env"
    export GATED_ATTN=headwise
    export EMBED_BITS=7
    export EMBED_CLIP_SIGMAS=15.0
    source "$REPO_ROOT/runs/configs/v2_fullstack.env"

    export SEED=$SEED
    export RUN_ID=x1_fullstack_seed${SEED}

    echo "  RUN_ID:              $RUN_ID"
    echo "  SEED:                $SEED"
    echo "  GRAD_ACCUM_STEPS:    $GRAD_ACCUM_STEPS"
    echo "  TRAIN_BATCH_TOKENS:  $TRAIN_BATCH_TOKENS"
    echo "  EMA_DECAY:           $EMA_DECAY"
    echo "  PREQUANT_TTT_ENABLED: $PREQUANT_TTT_ENABLED"
    echo "  GATED_ATTN:          $GATED_ATTN"
    echo "============================================"
    echo ""

    torchrun --standalone --nproc_per_node=8 train_gpt_v2.py

    echo "==> Seed $SEED done. Log: logs/${RUN_ID}.txt"
done

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================"
echo "  X1 FULL STACK — 3-SEED RESULTS (8xH100)"
echo "============================================"

python3 -c "
import os, re, math

seeds = [42, 1337, 2025]
results = []

for seed in seeds:
    log_file = f'logs/x1_fullstack_seed{seed}.txt'
    if not os.path.isfile(log_file):
        print(f'  WARNING: {log_file} not found, skipping seed {seed}')
        continue
    text = open(log_file).read()

    ttt_bpb = ttt_loss = sw_bpb = preq_bpb = postpq_bpb = None

    m = re.search(r'quantized_ttt val_loss:([\d.]+) val_bpb:([\d.]+)', text)
    if m: ttt_loss = float(m.group(1)); ttt_bpb = float(m.group(2))

    m = re.search(r'quantized_sliding_window val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: sw_bpb = float(m.group(1))

    m = re.search(r'pre-quantization post-ema val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: preq_bpb = float(m.group(1))

    m = re.search(r'post-prequant-ttt val_loss:[\d.]+ val_bpb:([\d.]+)', text)
    if m: postpq_bpb = float(m.group(1))

    quant_bytes = total_bytes = None
    m = re.search(r'Serialized model quantized\+\w+: (\d+) bytes', text)
    if m: quant_bytes = int(m.group(1))
    m = re.search(r'Total submission size quantized\+\w+: (\d+) bytes', text)
    if m: total_bytes = int(m.group(1))

    steps = None
    for m_s in re.finditer(r'step (\d+) \|', text):
        steps = int(m_s.group(1))

    # Get eval time
    eval_time = None
    m = re.search(r'quantized_ttt.*eval_time:(\d+)ms', text)
    if m: eval_time = int(m.group(1))

    # Get train time
    train_time = None
    m = re.search(r'train_time: ([\d.]+)ms', text)
    if not m: m = re.search(r'train_time: ([\d.]+)m', text)
    if m: train_time = float(m.group(1))

    results.append({
        'seed': seed, 'ttt_bpb': ttt_bpb, 'ttt_loss': ttt_loss,
        'sw_bpb': sw_bpb, 'preq_bpb': preq_bpb, 'postpq_bpb': postpq_bpb,
        'quant_bytes': quant_bytes, 'total_bytes': total_bytes,
        'steps': steps, 'eval_time': eval_time,
    })

print()
print(f'  {\"Seed\":>6} | {\"Pre-Q\":>8} | {\"PostPQ\":>8} | {\"SW BPB\":>8} | {\"TTT BPB\":>9} | {\"Weights\":>12} | {\"Total\":>12} | Budget?')
print(f'  {\"-\"*6} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*9} | {\"-\"*12} | {\"-\"*12} | -------')

for r in results:
    pq = f\"{r['preq_bpb']:.4f}\" if r['preq_bpb'] else '?'
    ppq = f\"{r['postpq_bpb']:.4f}\" if r['postpq_bpb'] else '?'
    sw = f\"{r['sw_bpb']:.4f}\" if r['sw_bpb'] else '?'
    tb = f\"{r['ttt_bpb']:.4f}\" if r['ttt_bpb'] else '?'
    qb = f\"{r['quant_bytes']:,}\" if r['quant_bytes'] else '?'
    ttb = f\"{r['total_bytes']:,}\" if r['total_bytes'] else '?'
    budget = 'Yes' if r['total_bytes'] and r['total_bytes'] < 16_000_000 else 'No/?'
    print(f\"  {r['seed']:>6} | {pq:>8} | {ppq:>8} | {sw:>8} | {tb:>9} | {qb:>12} | {ttb:>12} | {budget}\")

bpbs = [r['ttt_bpb'] for r in results if r['ttt_bpb'] is not None]
losses = [r['ttt_loss'] for r in results if r['ttt_loss'] is not None]

if len(bpbs) >= 2:
    mean_bpb = sum(bpbs) / len(bpbs)
    std_bpb = math.sqrt(sum((x - mean_bpb)**2 for x in bpbs) / (len(bpbs) - 1))
    mean_loss = sum(losses) / len(losses)

    print(f'  {\"-\"*6} | {\"-\"*8} | {\"-\"*8} | {\"-\"*8} | {\"-\"*9} | {\"-\"*12} | {\"-\"*12} |')
    print(f'  {\"Mean\":>6} |          |          |          | {mean_bpb:>9.4f} |              |              |')
    print(f'  {\"Std\":>6} |          |          |          | {std_bpb:>9.5f} |              |              |')

    print()
    print(f'  --- For submission.json ---')
    print(f'  \"val_bpb\": {mean_bpb:.4f},')
    print(f'  \"val_bpb_std\": {std_bpb:.5f},')
    print(f'  \"seed_results\": {{')
    for r in results:
        qb = r['quant_bytes'] if r['quant_bytes'] else 0
        print(f'    \"{r[\"seed\"]}\": {{\"val_bpb\": {r[\"ttt_bpb\"]:.4f}, \"artifact_bytes\": {qb}}},')
    print(f'  }}')

    print()
    print(f'  === SUBMISSION READINESS ===')
    all_budget = all(r['total_bytes'] and r['total_bytes'] < 16_000_000 for r in results)
    print(f'  All artifacts under 16 MB:  {\"YES\" if all_budget else \"NO\"}')
    print(f'  3-seed mean BPB:           {mean_bpb:.4f}')
    print(f'  3-seed std:                {std_bpb:.5f}')
    print(f'  vs SOTA (1.0136):          {mean_bpb - 1.0136:+.4f}')
    print(f'  vs PG baseline (1.2244):   {mean_bpb - 1.2244:+.4f}')
    print(f'  vs C6 8xH100 (1.0805):     {mean_bpb - 1.0805:+.4f}')
    print(f'  vs X1 2xH100 (1.0591):     {mean_bpb - 1.0591:+.4f}')

    if mean_bpb < 1.0136 - 0.005:
        print(f'  VERDICT: SOTA RECORD — clears 0.005 nats threshold!')
    elif mean_bpb < 1.0136:
        print(f'  VERDICT: Beats SOTA but does not clear 0.005 nats threshold.')
    else:
        print(f'  VERDICT: Non-record submission (still valid if unique/interesting).')
print()
"

echo ""
echo "============================================"
echo "  Next: copy logs, prepare PR submission"
echo "  Log files: logs/x1_fullstack_seed{42,1337,2025}.txt"
echo "============================================"
