# ashray — parameter golf experiments

Working directory for our rank-4-based experiments on 2×H100.

## Layout

```
ashray/
├── parameter-golf/
│   ├── train_v1.py                   # rank 4 script (PR #1769), unmodified
│   ├── train_v2.py                   # (later) our first code-level additions on top of v1
│   ├── train_v3.py                   # (later) next iteration
│   ├── data/                         # datasets + CaseOps helpers (gitignored contents)
│   │   ├── prepare_caseops_data.py
│   │   ├── lossless_caps.py
│   │   ├── tokenizers/
│   │   │   └── fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
│   │   ├── fineweb10B_raw/           # raw HF docs (after SETUP)
│   │   └── datasets/                 # CaseOps shards + byte sidecar (after SETUP)
│   └── records/                      # per-run logs + artifacts (gitignored)
│       └── <RUN_NAME>/
│           ├── train.log
│           └── final_model.int6.ptz
└── runs/
    ├── configs/
    │   ├── SETUP.md                  # one-time runpod setup instructions
    │   ├── v1_base.env               # our baseline config (rank 4 + MIN_LR=0.10)
    │   └── ...                       # per-experiment override files (as we add them)
    ├── run_v1_baseline_2gpu.sh       # our baseline run (anchor for deltas)
    └── ...                           # one script per experiment
```

## Baseline config

`runs/configs/v1_base.env` is the anchor for all future delta experiments. It's
the rank 4 stack + `MIN_LR=0.10` (a rank-3-sourced, validated, no-code-change
addition). All future experiments measure deltas against this anchor.

## Running

On a fresh Runpod box:

```bash
# one-time setup — datasets + tokenizer + sanity checks
# read runs/configs/SETUP.md first, then follow it top-to-bottom

# then, our anchor run:
bash runs/run_v1_baseline_2gpu.sh
```

Each script writes logs + artifacts to `parameter-golf/records/<RUN_NAME>/`.

## Conventions

- **`train_vN.py`** lives at `parameter-golf/` top level, one file per "version"; each version is a self-contained trainer.
- **`runs/*.sh`** are the runnable entry points. They `source runs/configs/<config>.env`, then `torchrun train_vN.py`, then print a results summary.
- **`runs/configs/*.env`** are env-var files. `v1_base.env` holds rank 4's config; per-experiment files only list overrides.
- **Don't edit `train_v1.py`** — it's our anchor. New ideas go in `train_v2.py` etc.
- **Results** are extracted directly from the training log by the shell script's `python3` summary block (grep for `pre-quantization post-ema`, `quantized val_loss`, `quantized_ttt val_loss`, `Total submission size`).
