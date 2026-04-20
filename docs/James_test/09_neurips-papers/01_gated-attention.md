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

## Terminology Glossary

| Term | Plain English | In This Paper |
|------|--------------|---------------|
| **Sparsity** | Most values in a collection are zero (or near-zero). Like a classroom where only 3 out of 30 students raise their hand — the response is "sparse." | The sigmoid gate outputs are mostly near 0 or near 1, so most head outputs get nearly zeroed out. Only a few heads "speak up" for any given token. This is **input-dependent sparsity** — which heads are active changes based on what the model is looking at. |
| **Sigmoid** | A function that squashes any number into the range (0, 1). Large positive inputs → ~1, large negative inputs → ~0, zero → 0.5. | Used as the gate: one learnable parameter per head gets passed through sigmoid to produce the dimmer value. |
| **Non-linearity** | Any operation that isn't just "multiply by a constant and add." Linear operations can be collapsed together; non-linear ones can't. | The sigmoid between W_v and W_O prevents them from collapsing into a single matrix, preserving the expressiveness of both layers. |
| **Attention head** | Transformers split attention into multiple parallel "heads," each learning to focus on different patterns (e.g., one head might track syntax, another tracks meaning). | Each head gets its own independent gate — "head-specific gating." One head can shut off while another stays fully active. |
| **Attention sink** | The first token (BOS) receives disproportionately high attention, not because it's important, but because softmax forces the model to attend *somewhere*. | Gating eliminates this by letting heads output near-zero, so they don't need to dump unused attention on BOS. |
| **PPL (Perplexity)** | A score measuring how "surprised" the model is by text. Lower = better. A PPL of 10 means the model is, on average, choosing between 10 equally likely next words. | The paper reports up to 0.2 PPL reduction with gating — a meaningful improvement at scale. |
| **MMLU** | A benchmark of ~16,000 multiple-choice questions across 57 subjects (math, history, law, medicine, etc.). Higher = better. | Gated models score +2 points on MMLU compared to ungated baselines. |
| **RULER** | A benchmark specifically testing how well models handle long sequences (longer than what they were trained on). Higher = better. | Gated models score +10 points — the biggest win in the paper, because removing attention sinks helps length generalization. |
| **SDPA** | Scaled Dot-Product Attention — the core attention operation: softmax(QK^T / √d) × V. The standard way transformers compute "who should attend to whom." | G1 (the best gate position) is placed right after SDPA output, before the dense projection. |
| **MoE (Mixture of Experts)** | An architecture where only a subset of parameters are active for each input. A "router" picks which expert networks to use. Allows massive models with lower compute cost. | The paper tests on a 15B MoE model — gating helps even at this scale. |
| **Loss spike** | A sudden, sharp increase in training loss — the model temporarily "forgets" what it learned. Common in large-scale training and can destabilize or crash runs. | Gating nearly eliminates loss spikes, making training more stable and allowing higher learning rates. |
| **Linear collapse** | When two consecutive linear layers (y = Ax, then z = By) are mathematically equivalent to a single layer (z = BAx = Cx). You have twice the parameters but no extra capability. | The W_v → W_O path in standard attention suffers from this. The sigmoid gate breaks the collapse. |
| **Length generalization** | The ability of a model to perform well on sequences longer than what it saw during training (e.g., trained on 4K tokens, tested on 32K). | Gated models generalize much better because they don't rely on attention sinks, which are calibrated to training-time sequence lengths. |

---

## Open Questions / Things to Explore

- How does this interact with nanochat's MQA + QK normalization?
- Could gated attention improve Parameter Golf scores under the 16 MB constraint?
- What's the compute overhead of adding one sigmoid per head?
