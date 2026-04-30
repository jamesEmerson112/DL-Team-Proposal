#!/bin/bash
set -euo pipefail

# P2: LoRA TTT + CaseOps on P1a base (8×H100)
# Expected: ~13 min training + ~8-10 min eval (LoRA TTT)
# Compare: P1a SGD TTT = 1.0769, SOTA = 1.0611

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_DIR="${SCRIPT_DIR}/../parameter-golf"
cd "$PG_DIR"

echo "=== P2: LoRA TTT + CaseOps on P1a Base ==="
echo "GPU count: $(nvidia-smi -L | wc -l)"
echo ""

# --- Step 1: CaseOps data prep (if tokenizer model exists) ---
CASEOPS_TOKENIZER="data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
CASEOPS_DATA="data/datasets/fineweb10B_sp8192_caseops"

if [ -f "$CASEOPS_TOKENIZER" ] && [ ! -d "$CASEOPS_DATA" ]; then
    echo "=== CaseOps: Reprocessing FineWeb shards ==="
    echo "This takes ~30-60 min. Ctrl+C to skip (will use regular sp8192)."
    python prepare_caseops_data.py \
        --input-dir data/datasets/fineweb10B_sp8192 \
        --output-dir "$CASEOPS_DATA" \
        --tokenizer "$CASEOPS_TOKENIZER"
    echo "=== CaseOps: Done ==="
fi

# --- Step 2: Source config ---
source "${SCRIPT_DIR}/configs/p2_lora_caseops.env"

# Enable CaseOps if data exists
if [ -d "$CASEOPS_DATA" ]; then
    export DATASETS_DIR="$CASEOPS_DATA"
    export TOKENIZER_PATH="$CASEOPS_TOKENIZER"
    echo "CaseOps: ENABLED (data at $CASEOPS_DATA)"
else
    echo "CaseOps: DISABLED (no data at $CASEOPS_DATA, using regular sp8192)"
fi

echo "Config: LoRA TTT rank=${TTT_LORA_RANK}, lr=${TTT_LORA_LR}, chunk=${TTT_LORA_CHUNK_SIZE}"
echo "SGD TTT: ${TTT_ENABLED}"
echo ""

# --- Step 3: Run ---
NGPUS="${NGPUS:-8}"
torchrun --standalone --nproc_per_node="$NGPUS" train_gpt.py

echo ""
echo "=== P2 Complete ==="
echo "Check logs/ for val_bpb results."
echo "Compare: P1a SGD TTT = 1.0769, SOTA = 1.0611"

# --- Summary extraction ---
LOGFILE=$(ls -t logs/*.txt 2>/dev/null | head -1)
if [ -n "$LOGFILE" ]; then
    echo ""
    echo "=== Results from $LOGFILE ==="
    grep -E "val_loss|val_bpb|ttt_lora|model_params|stopping|submission" "$LOGFILE" | tail -20
fi
