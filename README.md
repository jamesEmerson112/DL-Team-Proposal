# DL Team Proposal

Research and deep dive into Andrej Karpathy's LLM training frameworks: **nanoGPT**, **llm.c**, and **nanochat**.

## Focus: nanochat Training Pipeline

nanochat is a full end-to-end pipeline for building a ChatGPT-style model (~$100, ~4 hours on 8xH100).

### The 4 Training Stages

| Stage | Job | Dataset | Analogy |
|---|---|---|---|
| 1. Tokenizer | Build vocabulary | ClimbMix | Learning the alphabet |
| 2. Pretrain | Learn language | FineWeb-EDU (100B tokens) | Reading millions of books |
| 3. SFT | Learn to chat | SmolTalk + MMLU + GSM8K + SpellingBee (568K examples) | Practicing conversations with a tutor |
| 4. RL (optional) | Get better at math | GSM8K | Doing homework and getting graded |

Each stage has its own dataset because each stage teaches the model something different. The model carries forward everything it learned — it's cumulative.

```
Raw Text ──▶ [Tokenizer] ──▶ [Pretrain] ──▶ [SFT] ──▶ [RL] ──▶ Chat UI
              ~30 min        ~2.5-3 hrs    ~8-30min   ~1hr
```

### Deep Dive Notes

Individual research notes are in `docs/<name>_notes/`:

- [00 — Table of Contents](docs/James_notes/00_table-of-contents.md) — start here
- [01 — Key Terms](docs/James_notes/01_key-terms.md) — glossary (BPE, SFT, RLHF, etc.)
- [02 — Comparison Table](docs/James_notes/02_comparison-table.md) — nanoGPT vs llm.c vs nanochat
- [03 — nanoGPT Notes](docs/James_notes/03_nanogpt-notes.md) — nanoGPT research
- [04 — llm.c Notes](docs/James_notes/04_llmc-notes.md) — llm.c research
- [05 — nanochat Notes](docs/James_notes/05_nanochat-notes.md) — nanochat research
- [06 — Datasets & Benchmarks](docs/James_notes/06_datasets-benchmarks-comparison.md) — datasets & benchmarks across all 3
- [07 — Training Stages](docs/James_notes/07_training-stages.md) — nanochat pipeline deep dive
- [08 — Parameter Golf](docs/James_notes/08_parameter-golf.md) — OpenAI's 16 MB model compression challenge
- [09 — NeurIPS Papers](docs/James_notes/09_neurips-papers.md) — 5 relevant papers for Related Work

### Parameter Golf (Competition)

We're participating in [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) — train the best LM in 16 MB / 10 min on 8xH100.

- [Overview & Setup](docs/parameter-golf/00_overview.md)
- [Experiments Tracker](docs/parameter-golf/experiments.md)
- [Findings & Insights](docs/parameter-golf/findings.md)
