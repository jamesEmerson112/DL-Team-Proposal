# Draft: Approach (What We Will Do)

> **Status:** Draft v1 — ready for team review
> **Section:** Project proposal "Approach" section
> **Constraint:** 4-5+ sentences, must describe specific implementation and experiments (not just "download and run")

---

## Overview

We will fork nanochat — an open-source end-to-end LLM training framework (~3,500 lines of Python/PyTorch) — and adapt it to operate under OpenAI's Parameter Golf constraints (1,024-token vocabulary, 16 MB artifact limit, 10-minute training budget on 8x H100 GPUs). This adapted fork becomes our shared experimental platform: once the baseline tiny model trains successfully, each team member independently explores a different improvement axis — architectural changes, dataset variations, and post-training techniques — and we periodically synchronize findings, combining the techniques that produce the steepest drops in training and validation perplexity. The goal is to iteratively produce better learning curves (lower perplexity over training steps) and ultimately submit a competitive Parameter Golf entry.

---

## Phase 1 — Adapting nanochat for Parameter Golf (team-wide)

Parameter Golf requires a self-contained training script using a fixed 1,024-token SentencePiece tokenizer and pre-tokenized FineWeb shards, while nanochat natively uses its own 32K+ Rust-based BPE tokenizer and FineWeb-EDU data format. We will modify nanochat's tokenizer module (`nanochat/tokenizer.py`) to load the competition's SentencePiece model, update the data loader in `base_train.py` to read the competition's shard format, and configure the model architecture to fit within 16 MB (approximately 9 layers, 512 hidden dimensions, 4 KV heads). We will then train an unmodified baseline model and log its learning curves — training perplexity and validation perplexity over steps — as the reference point all subsequent experiments are measured against.

## Phase 2 — Parallel experimentation (split across 3 team members)

With the baseline established, each member pursues a different improvement direction:

- **Architectural modifications (gated attention).** Informed by the NeurIPS 2025 Best Paper, we will modify nanochat's transformer (`gpt.py`) to insert learnable gating scalars into the attention layers — testing gating at all layers versus only early layers (where attention sinks form) — and compare the resulting learning curves against the ungated baseline.
- **Dataset and training variations.** We will experiment with data ordering and shard selection within the competition's 8 billion available training tokens, as well as hyperparameter tuning (learning rate schedule, batch size, warmup/decay), to identify which configurations yield faster convergence.
- **Post-training stages.** We will test whether nanochat's supervised fine-tuning (on SmolTalk/MMLU/GSM8K) and GRPO reinforcement learning improve BPB when applied to the small model, and whether alternative fine-tuning datasets shift the learning curve differently.

Each member produces learning curves for their variants. We regularly compare results: techniques that lower perplexity get kept; those that don't get discarded.

## Phase 3 — Integration and submission

We combine the winning techniques from Phase 2 into a single best-configuration model, verify that the combined approach produces better learning curves than any individual technique alone, and package the result into a Parameter Golf submission targeting a BPB below the current baseline of 1.2244. The final submission is a single self-contained `train_gpt.py` flattened from our modified nanochat fork.

## Analysis and deliverables

The primary deliverable is a set of comparative learning curves — training perplexity and validation perplexity plotted over training steps — for every variant we test. The final analysis ranks each technique by how much it lowered the perplexity curves relative to the baseline, identifies which combinations compound and which conflict, and documents reproducible training recipes for each experiment. All modified code, logs, and our Parameter Golf submission are published as project artifacts.

---

## Key notes

- Phase 1 (nanochat adaptation) is the core implementation work — satisfies "more than download and run"
- The gated attention modification adds genuine architectural research
- Each experiment changes one variable at a time for causal attribution
- All experiments happen at Parameter Golf model size, keeping everything focused
- Post-training stages (SFT/GRPO) may or may not be feasible within the 10-min budget — that itself is a finding worth reporting
