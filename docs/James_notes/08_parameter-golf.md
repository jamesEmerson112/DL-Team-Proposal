# OpenAI Parameter Golf — Technical Reference

> OpenAI Model Craft Challenge: "Parameter Golf" — train the best LM that fits in 16 MB.

## What It Is

An open competition by OpenAI (March 18 – April 30, 2026) challenging participants to train the best language model under extreme constraints. Named after golf — lowest score wins.

- **Prize:** $1M in compute credits (via RunPod)
- **Recruitment angle:** OpenAI is using this to identify early-career researchers (undergrads, recent grads, Olympiad medalists)
- **Community:** Discord channels `#parameter-golf-discussions` and `#parameter-golf-announcements`

## Constraints

| Constraint | Limit |
|---|---|
| **Artifact size** | 16 MB (decimal: 16,000,000 bytes) — weights + training code combined |
| **Training time** | 10 minutes on 8×H100 (SXM) |
| **Network** | No external downloads or network calls during evaluation |
| **Self-contained** | Artifact must be fully reproducible without external dependencies |

## Dataset

**FineWeb** retokenized with a 1024-token BPE vocabulary (`sp1024` variant).

| Property | Value |
|---|---|
| Training tokens | 8B (80 shards, 100M tokens each) |
| Full dataset | 10B tokens (100 shards) |
| Validation | Fixed first 50K documents from FineWeb |
| Tokenizer | `fineweb_1024_bpe.model` (SentencePiece, 1024 vocab) |
| Reproducibility | SHA-256 manifest for document selection verification |

### Download

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
# Populates: ./data/datasets/fineweb10B_sp1024/
```

Use `--train-shards N` for smaller subsets during development.

## Evaluation Metric

**Bits Per Byte (BPB)** — tokenizer-agnostic compression on FineWeb validation.

- Logs both `val_bpb` and `val_loss` (in nats) during training
- BPB is the primary scoring metric (lower = better)
- **SOTA threshold:** new records must beat existing by ≥0.005 nats at p < 0.01 significance
- Must provide enough run logs to prove statistical significance

### How BPB relates to perplexity

| Metric | Formula | What it measures |
|---|---|---|
| Loss (nats) | Cross-entropy loss | Raw model performance |
| Perplexity | `exp(loss)` | "How many choices is the model confused between?" |
| BPB | `loss × tokens / bytes` | Compression efficiency (tokenizer-agnostic) |

BPB normalizes across different tokenizers — a model with vocab 1024 and a model with vocab 32K can be compared fairly.

## How to Run

### Training (remote, 8xH100)

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training (local, Apple Silicon)

```bash
# Smoke test with reduced shards
python3 train_gpt_mlx.py
```

### Key Environment Variables

| Variable | Purpose |
|---|---|
| `RUN_ID` | Name for the training run |
| `DATA_PATH` | Path to dataset shards |
| `TOKENIZER_PATH` | Path to SentencePiece tokenizer |
| `VOCAB_SIZE` | Vocabulary size (1024 for sp1024) |
| `ITERATIONS` | Number of training iterations |
| `TRAIN_BATCH_TOKENS` | Batch size in tokens |
| `VAL_LOSS_EVERY` | Validation frequency |
| `MAX_WALLCLOCK_SECONDS` | Training time limit |

## Baseline Model

| Property | Value |
|---|---|
| Layers | 9 |
| Dimensions | 512 |
| Vocabulary | 1024 |
| KV Heads | 4 |
| Embeddings | Tied |
| Compressed size | <16 MB |
| **Score (BPB)** | **1.2244** |

Non-record reference (4hr unlimited compute): 1.2074 BPB

## Submission Format

Submit a PR to `openai/parameter-golf` with a new folder in `/records/track_10min_16mb/`:

1. `README.md` — detailed explanation of approach
2. `submission.json` — name, GitHub ID, val_bpb, metadata
3. Training log (auto-generated)
4. `train_gpt.py` + dependencies (must execute within folder)

OpenAI verifies reproducibility before merging. Non-reproducible submissions are disqualified.

## Comparison: Parameter Golf vs Our nanochat Project

| Aspect | Parameter Golf | Our Project (nanochat) |
|---|---|---|
| **Goal** | Best compression in 16 MB | Full chat model pipeline |
| **Dataset** | FineWeb (sp1024, 1024 vocab) | FineWeb-EDU (32K–65K vocab) |
| **Hardware** | 8×H100 | 8×H100 |
| **Training stages** | Pretraining only | Pretrain + SFT + RLHF |
| **Model size** | <16 MB compressed | ~560M parameters |
| **Training time** | 10 minutes | ~4 hours |
| **Metric** | BPB (bits per byte) | Loss + perplexity (learning curves) |
| **Framework** | Custom `train_gpt.py` (PyTorch) | nanochat (PyTorch) |
| **Lineage** | modded-nanogpt speedruns | nanochat (successor to nanoGPT) |
| **Cost** | Free (RunPod credits) | ~$100 |

### Why it's relevant

1. **Same dataset family** — both use FineWeb
2. **Same hardware** — both target 8×H100
3. **Same community** — modded-nanogpt speedruns → nanochat → Parameter Golf
4. **Complementary focus** — we study the full pipeline; they study extreme compression
5. **Same metrics** — BPB and val_loss are directly related to the perplexity/loss curves we need

## Sources

- [OpenAI — Parameter Golf announcement](https://openai.com/index/parameter-golf/)
- [GitHub — openai/parameter-golf](https://github.com/openai/parameter-golf)
- [GitHub — parameter-golf dataset README](https://github.com/openai/parameter-golf/blob/main/data/README.md)
- [The Decoder — Parameter Golf writeup](https://the-decoder.com/openai-turns-model-compression-into-a-talent-hunt-with-its-16-mb-parameter-golf-challenge/)
