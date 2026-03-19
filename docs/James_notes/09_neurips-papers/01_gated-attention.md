# Paper 1: Gated Attention for Large Language Models — Study Notes

> NeurIPS 2025 — Best Paper Award (Main Track)
> arXiv:2505.06708 | [GitHub](https://github.com/qiuzh20/gated_attention)

---

## Background: What is Gating?

A gate is like a dimmer switch — a learned value between 0 and 1 that controls how much of a signal passes through. The network learns *when* to let information flow and *when* to suppress it.

**History of gating in neural networks:**

- **LSTMs (1997)** — forget/input/output gates controlling memory over time steps
- **Highway Networks (2015)** — gates deciding whether a layer transforms data or passes it through unchanged
- **GRUs (2017)** — simplified LSTM with fewer gates
- **Modern:** state-space models (Mamba), various attention variants — all use gating

**The problem:** everyone uses gating because it works, but nobody rigorously isolated *why* it works. Prior work (SwitchHeads, Native Sparse Attention) conflated gating with other architectural changes.

---

## Core Contribution: Where to Put the Gate?

Five positions tested inside a standard attention block:

| Position | Location | Result |
|----------|----------|--------|
| G1 | After scaled dot-product attention output | **Best overall** (up to -0.2 PPL, +2 MMLU) |
| G2 | After value projection | Second best, notable PPL improvement |
| G3 | After key projection | Moderate |
| G4 | After query projection | Moderate |
| G5 | After dense output layer | Least effective |

The gate itself is minimal: one sigmoid per attention head.

---

## Why G1 Works: Three Mechanisms

### 1. Non-Linearity (Breaking Linear Collapse)

Without a gate, the value projection (W_v) and dense output (W_O) are consecutive linear layers. Two linear operations compose into one — you're paying for two weight matrices but only getting the expressiveness of one.

The sigmoid gate between them is non-linear, so the two layers can no longer collapse. Both do independent useful work. More parameters = more expressiveness, as intended.

### 2. Sparsity

The sigmoid gate pushes many values close to 0 or 1, creating a bimodal distribution. This means for any given input, many attention head outputs get nearly zeroed out. The network learns an **input-dependent filter** — "for this token, only certain heads matter."

### 3. Eliminating Attention Sinks

**What's an attention sink?** Softmax forces attention weights to sum to 1. When a head has nothing useful to attend to, it can't output "nothing" — it dumps attention on the first token (BOS). This wastes capacity and breaks length generalization.

**With gating:** the head can set its gate near 0, effectively outputting nothing. No need to waste attention on BOS as a garbage dump.

**Result:** gated models show no attention sinks and score **+10 points on RULER** (long-context benchmark), demonstrating superior length generalization.

---

## Experimental Scale

- **Models:** 15B MoE and 1.7B dense
- **Training data:** 3.5 trillion tokens
- **Training stability:** gating nearly eliminates loss spikes, enabling larger learning rates

---

## Open Questions / Things to Explore

- How does this interact with nanochat's MQA + QK normalization?
- Could gated attention improve Parameter Golf scores under the 16 MB constraint?
- What's the compute overhead of adding one sigmoid per head?
