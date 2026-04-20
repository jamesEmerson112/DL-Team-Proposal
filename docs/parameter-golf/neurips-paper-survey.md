# NeurIPS Paper Survey for Parameter Golf

Techniques from NeurIPS 2023-2024 papers applicable to the Parameter Golf challenge
(17M param GPT, 16 MB artifact, 10 min on 8xH100, 1024-token vocab, FineWeb).

---

## Summary Table

| # | Paper | Venue | Key Technique | Applicability to Parameter Golf |
|---|-------|-------|---------------|--------------------------------|
| 1 | **Not All Tokens Are What You Need (Rho-1)** | NeurIPS 2024 Best Paper Runner-Up | Selective Language Modeling — score tokens with a reference model, train only on high-excess-loss tokens | Could skip low-information tokens in FineWeb batches, getting more "learning per step." Needs a small reference model (the baseline itself after a few steps could serve). Potentially large BPB improvement for same compute. |
| 2 | **Scaling Data-Constrained Language Models** | NeurIPS 2023 Outstanding Paper | Scaling laws for repeated data — up to 4 epochs of repeated data has negligible loss vs unique data | Validates that if the 10-min window limits how much FineWeb you see, repeating data 2-4x is fine. Train longer on same data without penalty. |
| 3 | **Scaling Laws with Vocabulary** | NeurIPS 2024 | Optimal vocab size scales with model size; most LLMs use too-small vocabularies | At 17M params, 1024 tokens may actually be near-optimal or even oversized. The paper's scaling law could confirm whether 1024 is right or if 512/2048 would be better. Vocab is a free knob. |
| 4 | **Resolving Discrepancies in Compute-Optimal Scaling** | NeurIPS 2024 Spotlight | Corrects Kaplan vs Chinchilla scaling law gap; derives optimal LR and batch size scaling laws; AdamW beta2 tuning is essential at low batch sizes | Directly useful: the paper provides formulas for optimal LR and batch size given compute budget. Also shows warmup tokens should equal model size N — so ~17M tokens warmup for a 17M model. |
| 5 | **Building on Efficient Foundations: Structured FFN Layers** | NeurIPS 2024 | Replace dense FFN with low-rank + block-diagonal matrices; 17% throughput boost; steeper scaling curves | Could replace the FFN with structured matrices: same model quality with 32% fewer FFN params and 1.35x training speed. Directly trades FFN params for more steps in 10 min. |
| 6 | **SwitchHead: MoE Attention** | NeurIPS 2024 | Mixture-of-experts in attention layer; computes up to 8x fewer attention matrices; 44% compute, 27% memory | Could dramatically speed up attention computation. At 17M params with GQA already in use, SwitchHead could stack on top for further speedup. More steps in 10 min. |
| 7 | **MoEUT: Mixture-of-Experts Universal Transformers** | NeurIPS 2024 | Layer sharing (depth recurrence) + MoE to compensate for reduced parameter count | Directly relevant to Parameter Golf's 16 MB cap. Share layers (fewer unique params = smaller artifact) but use MoE routing to maintain expressivity. Could cut model size while keeping quality. |
| 8 | **Dynamic Layer Tying** | ICLR 2024 | RL-based dynamic selection of which layers to tie during training; 75-87% parameter reduction | Extreme parameter reduction for the 16 MB constraint. If 9 layers can be collapsed to 2-3 unique layers + routing, the artifact shrinks dramatically while maintaining perplexity. |
| 9 | **OneBit / BitNet** | NeurIPS 2024 / arXiv 2023 | 1-bit or 1.58-bit weight quantization during training (QAT); BitLinear as drop-in replacement for nn.Linear | The artifact is int8+zlib compressed. Training with QAT at 1.58-bit could yield a much smaller compressed artifact (well under 16 MB), freeing budget for more parameters/layers. |
| 10 | **Compact Language Models via Pruning and Knowledge Distillation** | NeurIPS 2024 | Structured pruning (depth, width, attention, MLP) + KD retraining with <3% of original data | If a larger model can be trained first (say 50M), then pruned+distilled to 17M in the remaining time, the result may beat training 17M from scratch. Two-phase approach. |
| 11 | **MemoryFormer** | NeurIPS 2024 | Replace FFN linear layers with locality-sensitive hash lookups into memory tables; near-zero FLOPs for FFN | Radical FFN replacement. Could make each forward pass much cheaper (FFN is ~2/3 of FLOPs), allowing more steps in 10 min. The hash tables would need to fit in 16 MB though. |
| 12 | **MATES: Model-Aware Data Selection** | NeurIPS 2024 | Continuously adapt data selection to model's evolving preferences during pretraining; 2x gains over static selection, halves FLOPs to reach target loss | Online data selection from FineWeb could double training efficiency. The influence model is tiny and runs alongside the main model. |
| 13 | **Faster LLM Training with Variable Sequence Length Curriculum** | NeurIPS 2024 | Start with short sequences, gradually increase; reduces early-training compute waste | Short sequences early = faster steps early = more gradient updates in 10 min. Simple to implement: just sort/bucket FineWeb by length. |
| 14 | **Surge Phenomenon in Optimal LR and Batch Size** | NeurIPS 2024 | For Adam-style optimizers, optimal LR first rises then falls as batch size increases; provides tuning guidance | Critical for 8xH100 setup where large batch sizes are natural. Helps find the right LR for the actual batch size being used. |
| 15 | **Small Batch Size Training / Why Gradient Accumulation is Wasteful** | NeurIPS 2025 | Small batch sizes are stable with proper Adam hyperparameter scaling; avoiding gradient accumulation improves per-FLOP performance | If currently using gradient accumulation, removing it and training with the natural per-GPU batch size could improve efficiency. Scale Adam beta2 to match. |
| 16 | **Why Warmup the Learning Rate?** | NeurIPS 2024 | Warmup forces network to well-conditioned loss landscape regions; optimal warmup tokens ~ model size N | For 17M model: warmup should be ~17M tokens. Too much warmup wastes precious steps; too little causes instability. |
| 17 | **Cross-Layer Attention (CLA)** | NeurIPS 2024 | Share KV heads between adjacent layers; 2x KV cache reduction with minimal accuracy loss | Reduces parameter count for KV projections across layers. Combined with GQA already in use, this could further shrink the model while maintaining quality. |
| 18 | **Are Emergent Abilities a Mirage?** | NeurIPS 2023 Outstanding Paper | Apparent emergence is an artifact of discontinuous metrics, not model behavior | Not directly actionable for BPB optimization, but reinforces that BPB (a continuous metric) is the right evaluation target — small models improve smoothly, no "emergence threshold" to worry about. |

---

## Top 5 Most Actionable Techniques (Ranked by Impact/Effort)

### 1. Rho-1 Selective Token Training
**Impact: HIGH | Effort: MEDIUM**
Train a small reference model first (or use the model's own early checkpoint), score FineWeb tokens by excess loss, mask out low-value tokens from the loss computation. The NeurIPS paper showed 30% improvement in math and 6.8% across general tasks. For BPB optimization, selectively weighting the loss on informative tokens could improve BPB per step significantly. Implementation: modify the loss function to weight tokens by their reference-model excess loss.

### 2. Structured FFN (Low-Rank + Block-Diagonal)
**Impact: HIGH | Effort: MEDIUM**
Replace the dense FFN with structured matrices. This gives 1.35x training speedup (more steps in 10 min) AND steeper scaling curves (better loss per FLOP). The FFN is typically 2/3 of transformer parameters — making it structured could allow a wider or deeper model within the same 16 MB budget.

### 3. Layer Tying / Depth Recurrence (MoEUT style)
**Impact: HIGH | Effort: HIGH**
Share weights across layers but add lightweight per-layer routing or adapters. A 9-layer model with 3 unique layer groups would have ~1/3 the parameters but similar effective depth. The freed parameter budget goes to wider dimensions or more heads. This is the single biggest lever for the 16 MB constraint.

### 4. Variable Sequence Length Curriculum
**Impact: MEDIUM | Effort: LOW**
Start training with short sequences (128 or 256 tokens), gradually increase to full context. Early steps are much faster (attention is O(n^2)), giving more gradient updates in the first minutes. Very easy to implement — just a data loader change.

### 5. Optimal LR/Batch Size from Scaling Laws
**Impact: MEDIUM | Effort: LOW**
Use the NeurIPS 2024 scaling law corrections to compute the right LR and batch size for 17M params on 8xH100. The "surge phenomenon" paper shows Adam's optimal LR is non-monotonic with batch size — this is likely undertrained. Also set warmup tokens = 17M (model size).

---

## Techniques Already in Use (Confirmed)

- **GQA (Grouped Query Attention)** — already reduces KV parameters
- **RoPE** — standard positional encoding
- **RMSNorm** — efficient normalization
- **ReLU^2** — activation function
- **Muon optimizer** — second-order optimizer, already ~1.4x faster than AdamW
- **int8 + zlib compression** — for the 16 MB artifact constraint

## Techniques to Investigate Further

- **BitNet / QAT at 1.58-bit**: Could the model be trained at lower precision, yielding a smaller compressed artifact and freeing parameter budget?
- **Cross-Layer Attention + GQA**: Stacking CLA on top of GQA for maximum KV parameter sharing
- **MemoryFormer hash tables**: Radical but could be transformative if hash tables compress well under zlib
- **Muon -> NorMuon upgrade**: 11% improvement over Muon with same memory footprint (NeurIPS 2025 workshop paper)

---

## Sources

- [NeurIPS 2024 Best Paper Awards](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/)
- [NeurIPS 2023 Paper Awards](https://blog.neurips.cc/2023/12/11/announcing-the-neurips-2023-paper-awards/)
- [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- [Scaling Laws with Vocabulary](https://neurips.cc/virtual/2024/poster/93395)
- [Resolving Discrepancies in Compute-Optimal Scaling](https://arxiv.org/abs/2406.19146)
- [Structured Feedforward Layers for LLMs](https://arxiv.org/abs/2406.16450)
- [SwitchHead: MoE Attention](https://arxiv.org/abs/2312.07987)
- [MoEUT: MoE Universal Transformers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/321387ba926b8e58d3591c0aeb52ffc2-Abstract-Conference.html)
- [Dynamic Layer Tying](https://arxiv.org/abs/2401.12819)
- [OneBit: Extremely Low-bit LLMs](https://neurips.cc/virtual/2024/poster/94602)
- [Compact LMs via Pruning and KD](https://arxiv.org/abs/2407.14679)
- [MemoryFormer](https://arxiv.org/abs/2411.12992)
- [MATES: Model-Aware Data Selection](https://neurips.cc/virtual/2024/poster/96504)
- [Variable Sequence Length Curriculum](https://proceedings.neurips.cc/paper_files/paper/2024/file/3f9bf45ea04c98ad7cb857f951f499e2-Paper-Conference.pdf)
- [Surge Phenomenon in LR and Batch Size](https://neurips.cc/virtual/2024/poster/94086)
- [Small Batch Size Training](https://neurips.cc/virtual/2025/poster/119899)
- [Why Warmup the Learning Rate](https://neurips.cc/virtual/2024/poster/95431)
- [Cross-Layer Attention](https://neurips.cc/virtual/2024/poster/95548)
- [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
- [NorMuon](https://arxiv.org/abs/2510.05491)
