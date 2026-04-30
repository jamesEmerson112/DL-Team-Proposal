# One-time setup on Runpod

Run this once per fresh pod, before any `run_vN_*.sh` script. Populates `parameter-golf/data/`.

**Disk requirement:** you need at least **100 GB** on `/workspace`. The intermediate
docs_selected.jsonl is 48 GB on its own, and the full pipeline peaks at ~80 GB before
the raw docs can be deleted. A 50 GB pod is not enough.

All paths below assume your pod working dir is the ashray/ root (i.e. `pwd` ends in `ashray`).

## 0. Start tmux

Long-running downloads + prep will drop if your SSH session dies. Start a tmux session first:

```bash
tmux new -s pg
# If you get disconnected later: `tmux attach -t pg`
```

## 1. Python deps

```bash
pip install --upgrade pip
pip install huggingface-hub sentencepiece numpy brotli
```

The Runpod Parameter Golf template already has PyTorch + FlashAttention 3 pre-installed.

Quick import check:

```bash
python3 -c "import huggingface_hub, sentencepiece, numpy, brotli; print('deps OK')"
```

## 2. Download docs_selected.jsonl + vanilla SP8192 shards via the official tool

```bash
python3 parameter-golf/data/cached_challenge_fineweb.py --variant sp8192 --with-docs
```

This uses `openai/parameter-golf`'s official downloader. It:

- Resolves HF cache symlinks correctly (this is the fix for the earlier symlink bug we hit).
- Downloads the vanilla SP8192 pre-tokenized shards (~16 GB).
- Downloads the vanilla SP8192 tokenizer model (`fineweb_8192_bpe.model`).
- With `--with-docs`, also downloads `docs_selected.jsonl` (~48 GB, the raw text corpus).

Output layout:

```
parameter-golf/data/
├── docs_selected.jsonl                     # 48 GB raw text (needed for step 3)
├── docs_selected.source_manifest.json
├── datasets/
│   └── fineweb10B_sp8192/
│       ├── fineweb_train_000000.bin ... fineweb_train_000079.bin  (80 shards, unused by our CaseOps pipeline)
│       └── fineweb_val_000000.bin
└── tokenizers/
    └── fineweb_8192_bpe.model              # vanilla SP8192 tokenizer (unused by CaseOps)
```

Takes ~5–20 minutes depending on pod network.

Verify:

```bash
ls -lh parameter-golf/data/docs_selected.jsonl
# Should show ~48 GB
df -h /workspace
# Should show roughly 65 GB used (48 GB docs + 16 GB vanilla shards + ~1 GB repo/system)
```

If `df` shows >90 GB used, the hardlink step fell back to a copy. Purge the HF cache to reclaim:

```bash
rm -rf ~/.cache/huggingface/
df -h /workspace
```

## 3. Re-tokenize into CaseOps SP8192 shards + byte sidecar

```bash
cd parameter-golf/data
python3 prepare_caseops_data.py \
    --docs ./docs_selected.jsonl \
    --out  ./datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
cd ../..
```

What this does:

- Applies the bijective case transform to every doc in `docs_selected.jsonl`.
- Tokenizes with the CaseOps SP8192 model (prepends `BOS_ID=1` per doc).
- Writes `fineweb_train_*.bin` and `fineweb_val_*.bin` shards.
- Writes `fineweb_val_bytes_*.bin` sidecar — per-token original-UTF-8-byte counts used for BPB accounting.

Runtime: ~30–60 min single-machine, CPU-bound. Progress prints every 10K docs:

```
  processed 10000 docs  train_shards=0  val_shards=0
  processed 20000 docs  train_shards=0  val_shards=1
  processed 30000 docs  train_shards=1  val_shards=1
  ...
```

Output lands at
`parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`.
The nested `datasets/datasets/` is intentional — that's what `train_v1.py` expects when `CASEOPS_ENABLED=1` and `DATA_DIR=./data`.

## 4. Sanity check

Verify BOS markers land in the val shard (TTT's `_find_docs` requires them) and that token/byte
shard counts align:

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path
root = Path("parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved")
d = np.fromfile(root / "fineweb_val_000000.bin", dtype=np.uint16)
tokens = d[512:]  # skip 256 int32 header = 512 uint16 slots
bos = int((tokens == 1).sum())
print(f"val shard tokens: {tokens.size:,}")
print(f"BOS markers:      {bos:,}  (must be > 0)")
assert bos > 0, "prep_caseops_data.py did not prepend BOS per doc"

tok_files = sorted(p for p in root.glob("fineweb_val_0*.bin") if "_bytes_" not in p.name)
byte_files = sorted(root.glob("fineweb_val_bytes_*.bin"))
print(f"val tokens shards: {len(tok_files)}")
print(f"val bytes shards:  {len(byte_files)}")
print(f"train shards:      {len(list(root.glob('fineweb_train_*.bin')))}")
assert len(tok_files) == len(byte_files), "token and byte shard counts must match"
for t, b in zip(tok_files, byte_files):
    assert t.stat().st_size == b.stat().st_size, f"size mismatch: {t.name} vs {b.name}"
print("shards aligned ✓")
PY
```

## 5. Reclaim disk (optional but recommended)

After CaseOps prep succeeds, the raw docs and vanilla SP8192 shards are no longer needed:

```bash
rm -f parameter-golf/data/docs_selected.jsonl
rm -f parameter-golf/data/docs_selected.source_manifest.json
rm -rf parameter-golf/data/datasets/fineweb10B_sp8192/
rm -f parameter-golf/data/tokenizers/fineweb_8192_bpe.model
df -h /workspace
# Should free ~64 GB.
```

Keep these in place if you want to later run vanilla-SP8192 comparisons.

## 6. GPU check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

Expect 2× `NVIDIA H100 80GB HBM3`. If you see more than 2 and want to pin to 2, prefix your training commands with `CUDA_VISIBLE_DEVICES=0,1`.

---

Done. You can now run any `runs/run_v*_*.sh` script. Start with:

```bash
bash runs/run_v1_baseline_2gpu.sh
```
