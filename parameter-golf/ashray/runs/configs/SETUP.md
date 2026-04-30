# One-time setup on Runpod

Run this once per fresh pod, before any `run_vN_*.sh` script. Populates `parameter-golf/data/`.

All paths below assume your pod working dir is the ashray/ root (i.e. `pwd` ends in `ashray`).

## 1. Python deps

```bash
pip install --upgrade pip
pip install huggingface-hub sentencepiece numpy brotli
# FlashAttention 3: comes pre-installed on the Runpod Parameter Golf template.
# If missing, follow the install note in the rank 4 PR README.
```

## 2. Download raw FineWeb docs from HuggingFace

The CaseOps prep script needs the pre-tokenization doc stream.

```bash
mkdir -p parameter-golf/data/fineweb10B_raw

python3 - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

out = Path("parameter-golf/data/fineweb10B_raw")
out.mkdir(parents=True, exist_ok=True)

for fname in ("docs_selected.jsonl", "docs_selected.source_manifest.json"):
    cached = hf_hub_download(
        repo_id="willdepueoai/parameter-golf",
        filename=fname,
        subfolder="datasets",
        repo_type="dataset",
    )
    dst = out / fname
    if dst.exists():
        dst.unlink()
    try:
        os.link(cached, dst)
    except OSError:
        shutil.copy2(cached, dst)
    print(f"got {fname}: {dst.stat().st_size:,} bytes")
PY
```

Takes ~5–20 min depending on pod network.

## 3. Run CaseOps tokenization + byte sidecar generation

```bash
cd parameter-golf/data
python3 prepare_caseops_data.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
cd ../..
```

Takes ~30–60 min CPU-bound. Progress prints every 10K docs.

Output lands at
`parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`.
The nested `datasets/datasets/` is intentional — that's what `train_v1.py` expects when `CASEOPS_ENABLED=1` and `DATA_DIR=./data`.

## 4. Sanity check

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

# Check byte sidecar alignment
tok_files = sorted(p for p in root.glob("fineweb_val_0*.bin") if "_bytes_" not in p.name)
byte_files = sorted(root.glob("fineweb_val_bytes_*.bin"))
assert len(tok_files) == len(byte_files), "token and byte shard counts must match"
for t, b in zip(tok_files, byte_files):
    assert t.stat().st_size == b.stat().st_size, f"size mismatch: {t.name} vs {b.name}"
print(f"val shards aligned: {len(tok_files)} pairs")
PY
```

## 5. GPU check

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

Expect 2× `NVIDIA H100 80GB HBM3`. If you see more than 2 and want to pin to 2, prefix your training commands with `CUDA_VISIBLE_DEVICES=0,1`.

---

Done. You can now run any `runs/run_v*_*.sh` script. Start with:

```bash
bash runs/run_v1_baseline_2gpu.sh
```
