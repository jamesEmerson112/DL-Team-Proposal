# nanochat Training Stages — Deep Dive

## Overview

nanochat has 4 training stages. Each stage has a different job and its own dataset.
The model carries forward everything it learned from previous stages — it's cumulative.

| Stage | Job | Dataset | Analogy |
|---|---|---|---|
| 1. Tokenizer | Build vocabulary | ClimbMix | Learning the alphabet |
| 2. Pretrain | Learn language | FineWeb-EDU (100B tokens) | Reading millions of books |
| 3. SFT | Learn to chat | SmolTalk + MMLU + GSM8K + SpellingBee (568K examples) | Practicing conversations with a tutor |
| 4. RL (optional) | Get better at math | GSM8K | Doing homework and getting graded |

## Why Each Stage Needs Its Own Data

- You wouldn't use conversation data for pretraining — it's too small (568K examples vs 100B tokens)
- You wouldn't use raw web text for SFT — it's not in a conversation format
- Each stage's data is specifically shaped for what that stage needs to teach

## Pipeline Visualization

```
                         NANOCHAT TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════

  Raw Text                                                   Chat UI
  (web data)                                                 (you talk
   │                                                          to it)
   ▼                                                            ▲
┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌────┐    ┌──────┐
│  STAGE 1 │    │   STAGE 2    │    │ STAGE 3 │    │ S4 │    │INFER │
│          │    │              │    │         │    │    │    │      │
│TOKENIZER │───▶│  PRETRAIN    │───▶│  SFT    │───▶│ RL │───▶│ CHAT │
│          │    │              │    │         │    │    │    │      │
│ Build    │    │ Learn        │    │ Learn   │    │Rewrd│    │ CLI/ │
│ vocab    │    │ language     │    │ to chat │    │math │    │ Web  │
└──────────┘    └──────────────┘    └─────────┘    └────┘    └──────┘
   ~30min          ~2.5-3hrs          ~8-30min      ~1hr
                                                  (optional)
```

## Stage 1 — Tokenizer (~30 min)

**Job:** Convert raw text into numbers the model can process.

**Dataset:** ClimbMix (8 parquet shards, ~800MB)

**What it does:**
- Trains a custom Rust-based BPE tokenizer from scratch
- Builds a vocabulary of 32,768 or 65,536 tokens
- Achieves ~4.8x compression ratio (5 characters ≈ 1 token)
- Includes special tokens for conversation formatting and tool use

**Output:** A tokenizer that can encode/decode text ↔ token IDs

---

## Stage 2 — Pretrain (~2.5–3 hours)

**Job:** Teach the model to understand and generate language.

**Dataset:** FineWeb-EDU-100B (1,822 shards, ~100MB each)
- Curated educational web text
- depth-20 model uses 240 shards
- Alt: NVIDIA ClimbMix-400B for speedrun leaderboard entries

**What it does:**
- Standard next-token prediction on massive text
- Distributed across 8 GPUs via torchrun
- BOS-aligned data loading (every sequence starts at a document boundary)
- Targets Chinchilla-optimal 20:1 token-to-parameter ratio

**Output:** A base model that can predict the next token (but can't chat yet)

---

## Stage 3 — SFT (~8–30 min)

**Job:** Teach the model to follow instructions and have conversations.

**Dataset:** 4 datasets mixed together (~568K examples total):

```
SFT Training Data
┌─────────────────────────────────────────┐
│  SmolTalk ████████████████████  460K    │  → conversations
│  MMLU     ██████████           100K    │  → knowledge/reasoning
│  GSM8K    █                      8K    │  → math
│  SpellingBee                     ?K    │  → additional tasks
└─────────────────────────────────────────┘
```

**What it does:**
- Adapts base model for multi-turn conversation
- Teaches tool use via special tokens (`<|python_start|>...<|python_end|>`)
- Padded conversation sequences with proper masking

**Output:** A chat model that can follow instructions and respond to prompts

---

## Stage 4 — RL / GRPO (~1 hour, optional)

**Job:** Improve the model's math ability using reinforcement learning.

**Dataset:** GSM8K (8K grade-school math problems)

**What it does:**
- Uses GRPO (Group Relative Policy Optimization) — a simplified RLHF variant
- Model generates multiple answers to each math problem
- Correct answers get rewarded, wrong ones get penalized
- Still experimental/in development

**Output:** A chat model that's better at math reasoning

---

## Monitoring Training Progress

Each stage (except tokenizer and inference) should log metrics every N steps:
- **Training loss** and **validation loss** — is the model converging?
- **Training perplexity** and **validation perplexity** — how "surprised" is the model? (`perplexity = exp(loss)`)
- **RL stage:** also log reward scores

Plot these as **learning curves** (metrics over epochs/steps). The TAs require these plots in the final paper as proof of training.

Watch for:
- Val loss diverging from train loss → overfitting
- Loss plateauing early → learning rate or data issues
- Loss spiking → gradient instability

## Total Pipeline

- **Time:** ~4 hours on 8xH100
- **Cost:** ~$100
- **Script:** `runs/speedrun.sh` chains all stages together
