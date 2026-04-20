# Running nanochat with multiple GPUs

**Author:** James Vo
**Date:** 2026-04-13
**Status:** Verified on RunPod 8×H100 SXM 80GB
**Baseline run:** `d3_golf` — 36M param depth-3 model, CORE 0.036, val_bpb 1.16

## What you'll accomplish

By following this note, a teammate new to nanochat can spin up a RunPod GPU node, clone this project, install dependencies, train a depth-3 baseline nanochat model, and get CORE benchmark numbers — end-to-end in under 30 minutes for under $15 of compute. The same flow is the launch point for the learning-curve experiments (depth sweep, SFT on/off, smaller-vocab tokenizer).

---

## 1. Prerequisites

- **GitHub** account. Our repo `jamesEmerson112/DL-Team-Proposal` is public, so a plain HTTPS clone works. If you fork or make it private, set up a PAT or SSH deploy key on the pod.
- **RunPod** account with credit. ~$15–25 is enough for a first full run with buffer for debugging.
- **wandb** account. Optional but strongly recommended — this project is about comparing training curves across variants, which is what wandb's "compare runs" view is built for. Free tier is fine.
- **Network volume**: NOT required. The volume disk (below) is sufficient for single-pod work.

---

## 2. RunPod pod configuration

### GPU choice

| Config | Use case | On-demand ~$/hr |
|---|---|---|
| **8×H100 SXM 80GB** (primary) | Full speedrun, multi-GPU training | ~$24/hr |
| **8×A100 80GB** (fallback) | If H100 unavailable | ~$16/hr |
| **1×H100 80GB** | Quick d≤6 iteration runs | ~$3/hr |

For the d=3 baseline in this note, 8×H100 is overkill but gives fast iteration. A single H100 would also work with gradient accumulation.

### Storage config (this is where everyone gets confused)

| Disk | Lifecycle | Our size | Purpose |
|---|---|---|---|
| **Container disk** | Erased on pod *stop* | 80GB | OS, pip packages, scratch |
| **Volume disk** | Erased on *terminate*, survives *stop* | 150GB at `/workspace` | Data, tokenizer, checkpoints |
| **Network volume** | Survives everything, attachable across pods | N/A | Skip — optional |

**Rule of thumb**: anything that takes more than 10 min to recreate goes on the volume disk. Volume disk at 150GB × $0.10/GB/mo running is ~$15/mo, negligible vs GPU cost.

### Template

Pick the stock **"RunPod PyTorch 2.4"** template (Ubuntu 22.04, CUDA 12.x). nanochat uses `uv` for deps so a clean CUDA base is all you need.

### Port exposure

Expose **port 8000** in pod settings for the chat web UI later (only needed after SFT).

---

## 3. Environment setup — exact commands

After you've SSH'd or opened the web terminal on the pod, run these in order.

```bash
# 1) Install Python dev headers — CRITICAL for torch.compile / Triton JIT
#    Without this you'll hit `fatal error: Python.h: No such file or directory`
apt-get update && apt-get install -y python3.10-dev screen

# 2) Persistent env vars — route data + HF cache to volume disk so they survive stops
export NANOCHAT_BASE_DIR=/workspace/.cache/nanochat
export OMP_NUM_THREADS=1
export HF_HOME=/workspace/.cache/huggingface
echo 'export NANOCHAT_BASE_DIR=/workspace/.cache/nanochat' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=1'                           >> ~/.bashrc
echo 'export HF_HOME=/workspace/.cache/huggingface'       >> ~/.bashrc
mkdir -p $NANOCHAT_BASE_DIR $HF_HOME

# 3) Clone our project repo with the nanochat submodule
cd /workspace
git clone --recurse-submodules https://github.com/jamesEmerson112/DL-Team-Proposal.git
cd DL-Team-Proposal/nanochat

# 4) Install uv and the project's Python dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra gpu
source .venv/bin/activate

# 5) Upgrade wandb — CRITICAL
#    The pinned version rejects the new `wandb_v1_...` token format
uv pip install -U wandb

# 6) Log in to wandb (grab your key at https://wandb.ai/authorize)
wandb login

# 7) Start a persistent screen session so SSH disconnects don't kill training
#    Detach with Ctrl-A then D; reattach with `screen -r train`
screen -S train
```

Sanity check GPUs are visible:

```bash
nvidia-smi    # should list 8× H100 80GB, 0% util
```

---

## 3.5. wandb setup (what it is, how to get the key)

### What wandb is

**Weights & Biases** (wandb) is a hosted experiment-tracking service — think "Google Analytics for ML training runs." When nanochat reports a metric (loss, learning rate, GPU utilization, val_bpb, CORE score, etc.), wandb logs it to their cloud in real time. You open a browser and see live plots.

Why it matters for *this* project specifically: our goal is comparing learning curves across model variants (depth sweep, SFT on/off, smaller-vocab tokenizer). wandb's "compare runs" view overlays multiple runs on the same chart — you select `d12_golf`, `d4_golf`, `d6_golf`, hit Compare, and get a scaling-curve plot for free. Without wandb you'd be plotting from logs manually.

Free tier is generous (100GB storage, unlimited personal projects). Academic/personal use is covered.

### Sign up and grab the API key

1. Go to https://wandb.ai/signup and create an account (GitHub login is fastest).
2. Visit https://wandb.ai/authorize and copy the API key shown.
3. Note: the key format is `wandb_v1_XigH...` (86 chars). This is the current format — older wandb versions expected a 40-char legacy hex key, which is why we upgrade wandb in §3 step 5.
4. **Treat the key like a password.** Do not paste it into `.env` files that might be committed, chat logs, or screenshots. If exposed, rotate it at https://wandb.ai/settings.

### Log in on the pod

```bash
wandb login
# paste the 86-char wandb_v1_... key when prompted
# saves to ~/.netrc (on container disk — you'll re-login on each fresh pod)
```

Alternatively, set an env var so you don't have to re-run `wandb login` every pod:

```bash
export WANDB_API_KEY="wandb_v1_...your_key..."
echo 'export WANDB_API_KEY=wandb_v1_...' >> ~/.bashrc
```

### Launching runs with wandb

Set `WANDB_RUN=<name>` in front of the training command. This tags the run in the dashboard:

```bash
WANDB_RUN=d3_golf torchrun ... -m scripts.base_train -- --run=d3_golf ...
```

Name runs consistently so you can compare them later. Recommended scheme for this project:

- `d12_baseline` — pretrained only
- `d12_sft` — pretrained + SFT
- `d12_sft_rl` — pretrained + SFT + RL
- `d16_baseline`, `d20_baseline` — different depths
- `d12_vocab4096` — shrunk-vocab variant

### Watching the run

Dashboard lives at `https://wandb.ai/<your-username>/nanochat`. Key panels:

| Panel | What it tells you |
|---|---|
| `train/loss` | Loss curve. Should drop steadily; a flat line means learning stalled. |
| `val_bpb` | Validation bits-per-byte. Our primary scaling-curve metric. |
| `core_metric` | DCLM CORE score (logged at final step if `--core-metric-every=999999`). |
| `train/mfu` | Model FLOPS utilization. Low = small model or inefficient; 4.7% at d=3 is normal. |
| `train/tok_per_sec` | Training throughput. |
| GPU util / VRAM / temp | System panels (wandb auto-logs). Spot hardware issues. |

### If you can't or don't want to use wandb

Set `WANDB_MODE=disabled` before training:
```bash
export WANDB_MODE=disabled
```
Training works fine; you just lose the dashboard and have to pull metrics from stdout / the local `wandb/` directory.

---

## 4. Pipeline walkthrough — data → tokenizer → base model

### Three prereq commands (data + tokenizer)

```bash
# Downloads ~8 shards of FineWeb-EDU (~2B chars, ~800MB) from HuggingFace
python -m nanochat.dataset -n 8                    # 2–5 min

# Trains a BPE tokenizer at vocab=32,768 on the 2B chars
python -m scripts.tok_train                        # ~2 min, CPU-bound

# Sanity-checks the tokenizer; expect ~4.0–4.3 bytes/token compression
python -m scripts.tok_eval                         # <1 min
```

After these, the volume disk contains:

```
/workspace/.cache/nanochat/
├── data/              # 8 training shards
├── tokenizer/         # tokenizer.pkl, tokenizer.json
└── report/
```

### The golden d=3 baseline command

```bash
WANDB_RUN=d3_golf torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=3 \
    --device-batch-size=16 \
    --run="d3_golf" \
    --model-tag="d3_golf" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

**Wall time**: ~2 minutes total (30s compute + setup + final CORE eval).

**Why `--device-batch-size=16`**: at small depths the auto-tuned total batch (262K tokens) is smaller than one forward pass at the default `--device-batch-size=32`, and training aborts with an assertion. 16 works for d ≤ ~6.

**Why the `999999 / -1 / -1` flags**: they skip intermediate CORE evals, sample generations, and checkpoints during training, so the run is as fast as possible. CORE and the final checkpoint still run once at the end.

---

## 5. Gotchas & fixes

Every one of these bit us on the first run. Fixes are one-liners.

### 5.1 wandb `API key must be 40 characters long, yours was 86`

Symptom:
```
ValueError: API key must be 40 characters long, yours was 86
```

Cause: wandb migrated to a new `wandb_v1_...` 86-char token format, but the version pinned in `nanochat/uv.lock` still expects the legacy 40-char hex key.

Fix: upgrade wandb inside the venv.
```bash
uv pip install -U wandb
```

### 5.2 `FileNotFoundError: tokenizer.pkl`

Symptom:
```
FileNotFoundError: [Errno 2] No such file or directory:
  '/root/.cache/nanochat/tokenizer/tokenizer.pkl'
```

Cause: `base_train.py` tries to load the tokenizer at startup. It must already exist.

Fix: run data download + tokenizer training first.
```bash
python -m nanochat.dataset -n 8
python -m scripts.tok_train
```

### 5.3 `AssertionError: total_batch_size % world_tokens_per_fwdbwd == 0`

Symptom (after model config prints fine):
```
File ".../scripts/base_train.py", line 409, in <module>
    assert total_batch_size % world_tokens_per_fwdbwd == 0
AssertionError
```

Cause: the auto-tuner picks a small `total_batch_size` for tiny models (262K tokens for d=3), but the default `--device-batch-size=32` × 8 GPUs × 2048 seq gives 524K tokens/forward — larger than the entire batch.

Fix: reduce `--device-batch-size` so `device_batch_size × 8 × 2048` divides the auto-tuned `total_batch_size`. For d=3, use `--device-batch-size=16` (exact fit); for d=4–6, 16 also works.

### 5.4 `fatal error: Python.h: No such file or directory`

Symptom (during the CORE eval compile step):
```
/tmp/tmpXXX/cuda_utils.c:6:10: fatal error: Python.h: No such file or directory
    6 | #include <Python.h>
```

Cause: Triton JIT-compiles small CUDA kernels at runtime. The compile needs Python C headers (`python3.10-dev`), which aren't in most RunPod CUDA base images.

Fix:
```bash
apt-get update && apt-get install -y python3.10-dev
```

### 5.5 wandb interactive prompt hanging the run

Symptom: the training job stops after printing:
```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
```
and never progresses.

Cause: `wandb.init()` tries to prompt for an account choice, but the `torchrun` subprocess has no interactive stdin.

Fix (pick one):
```bash
# Option A — log in once (recommended)
wandb login

# Option B — skip wandb entirely
export WANDB_MODE=disabled
```

### 5.6 Data / tokenizer lost when pod terminates

Symptom: after a pod *terminate*, you have to re-download 800MB of data and re-train the tokenizer on every new pod.

Cause: nanochat defaults `NANOCHAT_BASE_DIR` to `$HOME/.cache/nanochat`, which lives on the container disk and dies with the pod.

Fix: point it at the volume disk.
```bash
export NANOCHAT_BASE_DIR=/workspace/.cache/nanochat
```
Put it in `~/.bashrc` so new shells inherit it (step 2 of §3 above does this).

---

## 6. Results + next steps

### 6.1 Our d=3 baseline (2026-04-13)

| Item | Value |
|---|---|
| wandb run | https://wandb.ai/vo-an-2503-georgia-institute-of-technology/nanochat/runs/4arf3flz |
| Checkpoint | `/workspace/.cache/nanochat/base_checkpoints/d3_golf/model_000492.pt` |
| Config | `n_layer=3, n_embd=256, n_head=2, n_kv_head=2, vocab=32768, seq_len=2048` |
| Total params | **35,913,808 (~36M)** |
| Embedding params | 33.5M (94% of total — wte 8.4M + value_embeds 16.8M + lm_head 8.4M) |
| Transformer params | 2.4M (6%) |
| Training tokens | 128.9M |
| Training time | 0.50 min wall (~30s compute) |
| Peak VRAM / GPU | 4.8 GB |
| MFU | 4.7% |
| **val_bpb** | **1.160** |
| **CORE metric** | **0.0361** (GPT-2 reference: 0.2565) |

#### Per-task eval snapshot (from the run log)

| Task | Accuracy | Random | Notes |
|---|---:|---:|---|
| hellaswag (10-shot) | 0.282 | 0.25 | marginally above chance |
| winogrande (0-shot) | 0.542 | 0.50 | decent signal |
| winograd (0-shot) | 0.524 | 0.48 | |
| lambada_openai (0-shot) | 0.140 | — | |
| boolq (10-shot) | 0.430 | 0.50 | below chance — model is confusing itself |
| squad (10-shot) | 0.056 | ~0.00 | near-zero QA ability |
| coqa (0-shot) | 0.038 | ~0.00 | near-zero QA ability |
| agi_eval_lsat_ar (3-shot) | 0.222 | 0.20 | |
| bigbench_cs_algorithms (10-shot) | 0.400 | — | |

### 6.2 Cost tracking for this run

| Line | Amount |
|---|---|
| Pod: 8×H100 SXM on-demand | ~$24/hr |
| Total pod time (including debug) | ~60 min |
| **Estimated spend** | **~$24** |
| Steady-state d=3 run after setup | ~$1 (~2 min) |
| Volume disk 150GB @ $0.10/GB/mo running | negligible for short runs |

### 6.3 How to interpret the numbers

- **val_bpb** (validation bits per byte) — continuous, vocab-invariant perplexity. **Lower is better.** Random = ~8.0, GPT-2 = 0.75. Use this for smooth scaling-curve plots across depth / training tokens.
- **CORE metric** — aggregate of 22 DCLM benchmark tasks, capability-weighted. **Higher is better.** Random ≈ 0.0, GPT-2 = 0.2565. Use for "does this model actually do anything useful" comparisons.
- **MFU** (model FLOPS utilization) — 4.7% is low *because d=3 is too small to saturate H100s*, not because anything is broken. Expect 40–50% at d=20+.
- **Peak VRAM 4.8 GB / GPU** — 6% of an H100's 80 GB. A d=3 run would fit on a consumer GPU.

### 6.4 Next-steps playbook (learning-curve project)

Three natural experiment tracks, all launched from this same setup:

**A) Depth sweep (scaling curve).**
Train d=2, d=4, d=6 with the same command (just change `--depth=` and the run/model tags). Plot CORE and val_bpb vs params or FLOPs. This is our primary learning-curve deliverable.
```bash
for D in 2 4 6; do
  WANDB_RUN=d${D}_golf torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
      --depth=$D --device-batch-size=16 \
      --run="d${D}_golf" --model-tag="d${D}_golf" \
      --core-metric-every=999999 --sample-every=-1 --save-every=-1
done
```

**B) Add SFT on top.**
Run supervised fine-tuning on the identity conversations dataset, then compare pre/post-SFT val_bpb and the chat-eval benchmarks (MMLU, GSM8K, HumanEval).
```bash
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --model-tag=d3_golf --device-batch-size=16 --run=d3_golf_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**C) Shrink to hit the 16MB Parameter Golf budget.**
At the current 32,768 vocab, embeddings alone are 33.5M params = 67MB in bf16 — we can't hit 16MB by shrinking depth alone. Retrain the tokenizer at a smaller vocab:
```bash
# Edit scripts/tok_train.py to set vocab_size=4096, then:
python -m scripts.tok_train
python -m scripts.tok_eval    # expect worse bytes/token; trade-off for smaller model
# re-run d=3 training
```
With vocab=4096, embedding params drop ~8× → ~4M params; total model ~6M params ≈ 12MB bf16. Hits budget. Add int8/int4 quantization for further headroom.
