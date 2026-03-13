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

- [comparison-table.md](docs/James_notes/comparison-table.md) — nanoGPT vs llm.c vs nanochat
- [nanogpt-notes.md](docs/James_notes/nanogpt-notes.md) — nanoGPT research
- [llmc-notes.md](docs/James_notes/llmc-notes.md) — llm.c research
- [nanochat-notes.md](docs/James_notes/nanochat-notes.md) — nanochat research
- [datasets-benchmarks-comparison.md](docs/James_notes/datasets-benchmarks-comparison.md) — datasets & benchmarks across all 3
- [key-terms.md](docs/James_notes/key-terms.md) — glossary (BPE, SFT, RLHF, etc.)
