# Parameter Golf — Getting Started

> Operational guide for participating in OpenAI's Parameter Golf competition.
> For research context, see [08_parameter-golf.md](../James_notes/08_parameter-golf.md).

## Quick Facts

| Item | Value |
|---|---|
| Competition | OpenAI Model Craft: Parameter Golf |
| Dates | March 18 – April 30, 2026 |
| Goal | Lowest BPB on FineWeb validation |
| Artifact limit | 16 MB (code + compressed weights) |
| Training time | 10 min on 8×H100 (SXM) |
| Prize | $1M in RunPod compute credits |
| GitHub | [openai/parameter-golf](https://github.com/openai/parameter-golf) |
| Discord | #parameter-golf-discussions, #parameter-golf-announcements |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

### 2. Download dataset

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
# Creates: ./data/datasets/fineweb10B_sp1024/ (8B tokens, 80 shards)
# Tokenizer: ./data/tokenizers/fineweb_1024_bpe.model
```

Use `--train-shards N` for smaller subsets during development.

### 3. Run baseline training

```bash
# Single GPU smoke test
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Full 8xH100 run
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Apple Silicon: use `train_gpt_mlx.py` for local testing.

### 4. Check your score

Training logs print `val_bpb` and `val_loss`. Baseline score: **1.2244 BPB**.

## Key Constraints

- No external downloads or network calls during evaluation
- Artifact must be self-contained and reproducible
- New SOTA must beat existing by ≥0.005 nats at p < 0.01
- "Spirit of the challenge" catch-all — don't try to game the rules

## Submission

PR to `openai/parameter-golf` → `/records/track_10min_16mb/`:
1. `README.md` — explain your approach
2. `submission.json` — name, GitHub ID, val_bpb
3. Training log
4. `train_gpt.py` + dependencies

## Ideas to Explore

- [ ] Attention mechanism changes (gated attention per NeurIPS 2025 paper)
- [ ] Data filtering/curriculum (which shards matter most?)
- [ ] Tokenizer optimization (1024 vocab is tiny — room for improvement?)
- [ ] Architecture search within 16 MB budget
- [ ] Weight compression/quantization strategies
- [ ] Training schedule optimization (learning rate, warmup, decay)
