#!/bin/bash
# =============================================================================
# SparseAttnGate + FusedSoftcapCE A/B on 2xH100 (4 runs)
#
# B0: base config
# B1: base + SparseAttnGate
# B2: base + FusedSoftcapCE
# B3: base + SparseAttnGate + FusedSoftcapCE
#
# Uses train_gpt_archive.py with explicit SP8192 paths.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_ROOT/parameter-golf"

cd "$PG_DIR"
echo "==> Working in $(pwd)"
mkdir -p logs

# Helper: base config
set_base() {
    # If you have a base env file you want to reuse, uncomment this:
    # source "$REPO_ROOT/runs/configs/v2_base.env"

    export DATA_PATH=./data/datasets/fineweb10B_sp8192
    export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
    export VOCAB_SIZE=8192

    # Keep this aligned with your simpler archive script defaults unless you want to tune them.
    export TRAIN_BATCH_TOKENS=524288
    export TRAIN_SEQ_LEN=1024
    export ITERATIONS=20000
    export WARMUP_STEPS=20
    export MAX_WALLCLOCK_SECONDS=600

    export NUM_LAYERS=9
    export MODEL_DIM=512
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=2
    export TIE_EMBEDDINGS=1
    export LOGIT_SOFTCAP=30.0
    export ROPE_BASE=10000.0
    export QK_GAIN_INIT=1.5

    export TIED_EMBED_LR=0.05
    export MATRIX_LR=0.04
    export SCALAR_LR=0.04
    export HEAD_LR=0.008
    export MUON_MOMENTUM=0.95
    export MUON_BACKEND_STEPS=5
    export BETA1=0.9
    export BETA2=0.95
    export ADAM_EPS=1e-8
    export GRAD_CLIP_NORM=0.0

    export ACTIVATION=relu2
    export GATED_ATTN=none
    export SLM_ENABLED=0
    export VALUE_RESIDUAL_ALPHA=0.0
    export QUANT_MODE=int8_zlib
    export SEED=1337

    # Default off; each run selectively enables them.
    export SPARSE_ATTN_GATE=0
    export SPARSE_ATTN_GATE_INIT=0.0
    export FUSED_SOFTCAP_CE=0
}

run_one() {
    local run_id="$1"
    local desc="$2"
    export RUN_ID="$run_id"

    echo ""
    echo "============================================"
    echo "  $desc"
    echo "  RUN_ID=$RUN_ID"
    echo "============================================"
    echo ""

    torchrun --standalone --nproc_per_node=2 train_gpt_archive.py
    echo "==> Done. Log: logs/${RUN_ID}.txt"
}

echo ""
echo "########################################"
echo "  SparseAttnGate + FusedSoftcapCE A/B"
echo "########################################"

# B0: Base
set_base
run_one "base_control" "B0: Base control"

# B1: SparseAttnGate only
set_base
export SPARSE_ATTN_GATE=1
export SPARSE_ATTN_GATE_INIT=0.0
run_one "base_sparse_only" "B1: Base + SparseAttnGate"

# B2: FusedSoftcapCE only
set_base
export FUSED_SOFTCAP_CE=1
run_one "base_fusedce_only" "B2: Base + FusedSoftcapCE"

# B3: Both
set_base
export SPARSE_ATTN_GATE=1
export SPARSE_ATTN_GATE_INIT=0.0
export FUSED_SOFTCAP_CE=1
run_one "base_sparse_fused" "B3: Base + SparseAttnGate + FusedSoftcapCE"

echo ""
echo "============================================"
echo "  SUMMARY"
echo "============================================"

python3 -c "
import os, re

runs = [
    ('B0 base',               'base_control'),
    ('B1 sparse only',        'base_sparse_only'),
    ('B2 fused CE only',      'base_fusedce_only'),
    ('B3 sparse + fused CE',  'base_sparse_fused'),
]

print()
print(f'{\"Run\":>22} | {\"Roundtrip BPB\":>13} | {\"Val BPB\":>10} | {\"Weights\":>12}')
print(f'{\"-\"*22} | {\"-\"*13} | {\"-\"*10} | {\"-\"*12}')

for label, run_id in runs:
    log_file = f'logs/{run_id}.txt'
    if not os.path.isfile(log_file):
        print(f'{label:>22} | {\"MISSING\":>13} | {\"\":>10} | {\"\":>12}')
        continue

    text = open(log_file).read()

    rt = None
    m = re.search(r'final_int8_zlib_roundtrip val_loss:[\\d.]+ val_bpb:([\\d.]+)', text)
    if m:
        rt = float(m.group(1))

    vb = None
    vals = re.findall(r'step:\\d+/\\d+ val_loss:[\\d.]+ val_bpb:([\\d.]+)', text)
    if vals:
        vb = float(vals[-1])

    qb = None
    m = re.search(r'Serialized model int8\\+zlib: (\\d+) bytes', text)
    if m:
        qb = int(m.group(1))

    rts = f'{rt:.4f}' if rt is not None else '?'
    vbs = f'{vb:.4f}' if vb is not None else '?'
    qbs = f'{qb:,}' if qb is not None else '?'
    print(f'{label:>22} | {rts:>13} | {vbs:>10} | {qbs:>12}')
"=========================="
echo "Logs:"
ls -1 logs/sidlike_*.txt 2>/dev/null || true
