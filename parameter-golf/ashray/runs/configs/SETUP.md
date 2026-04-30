# One-time setup on Runpod

Run this once per fresh pod, before any `run_vN_*.sh` script.

**What you get:** vanilla SP8192 tokenized shards (Kevin Clark's pre-export from HF repo
`kevclark/parameter-golf`). No CaseOps, no raw-docs download. Avoids the 48 GB
`docs_selected.jsonl` download and its CPU-heavy re-tokenization step.

**Disk requirement:** ~25 GB for the full 128-shard SP8192 export.

All paths below assume your pod working dir is the ashray/ root (i.e. `pwd` ends in `ashray`).

## 0. Install tmux + start a session

Long-running downloads will die if your SSH drops. Start tmux before anything else.

```bash
which tmux || apt-get update -qq && apt-get install -y -qq tmux
tmux new -s pg
```

Verify you're inside:

```bash
echo $TMUX
# Must print a path like /tmp/tmux-0/default,1234,0
# If empty, you are NOT in tmux — run `tmux new -s pg` again.
```

If your SSH drops later: reconnect to the pod, then `tmux attach -t pg`.

## 1. Python deps

```bash
pip install --upgrade pip
pip install huggingface-hub sentencepiece numpy brotli
python3 -c "import huggingface_hub, sentencepiece, numpy, brotli; print('deps OK')"
```

Runpod's Parameter Golf template has PyTorch + FlashAttention 3 pre-installed. Verify:

```bash
python3 -c "import torch; from flash_attn_interface import flash_attn_func; print(f'torch {torch.__version__}, FA3 OK')"
```

## 2. (Optional but strongly recommended) HuggingFace token

Unauthenticated HF downloads get rate-limited and disconnected on large files.
Create a free read-only token at <https://huggingface.co/settings/tokens>, then:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx   # paste yours
```

## 3. Download Kevin Clark's vanilla SP8192 shards

```bash
cd /workspace/DL-Team-Proposal/parameter-golf/ashray

# Per Kevin Clark's README: the default manifest.json (from willdepueoai/parameter-golf)
# doesn't include sp8192. Delete any cached copy so his manifest is fetched fresh.
rm -f parameter-golf/data/manifest.json

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 parameter-golf/data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

What this downloads (to `parameter-golf/data/`):

- `datasets/fineweb10B_sp8192/fineweb_train_{000000..000127}.bin` — 128 shards × ~200 MB = ~25 GB
- `datasets/fineweb10B_sp8192/fineweb_val_000000.bin` — ~200 MB
- `tokenizers/fineweb_8192_bpe.model` — ~370 KB
- `manifest.json`

Takes ~10–25 min depending on pod network. You should see HF download progress bars.

### If the download hangs or drops

- Make sure you're in tmux (`echo $TMUX`).
- HF token set (`echo ${HF_TOKEN:+set}` — should print `set`).
- Retry the same command: the HF downloader resumes from the local cache.

## 4. Sanity check

```bash
ls parameter-golf/data/datasets/fineweb10B_sp8192/ | wc -l
# Expect 129 (128 train shards + 1 val shard)

ls -lh parameter-golf/data/tokenizers/fineweb_8192_bpe.model
# Expect ~370 KB

df -h /workspace
# Expect ~25-30 GB used
```

Verify the val shard has BOS markers (needed by rank 4's phased-TTT eval):

```bash
python3 - <<'PY'
import numpy as np
d = np.fromfile("parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin", dtype=np.uint16)
tokens = d[512:]  # skip 256 int32 header = 512 uint16 slots
bos = int((tokens == 1).sum())
print(f"val shard tokens: {tokens.size:,}")
print(f"BOS markers:      {bos:,}  (must be > 0 for phased TTT)")
assert bos > 0
print("val shard OK ✓")
PY
```

If BOS count is 0, phased TTT's `_find_docs` will return no docs and eval will fail. The
shards from Kevin's HF repo should have BOS per doc. If they don't, tell me.

## 5. GPU check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

Expect 2× `NVIDIA H100 80GB HBM3`.

---

Done. You can now run any `runs/run_v*_*.sh` script. Start with:

```bash
bash runs/run_v1_baseline_2gpu.sh
```
