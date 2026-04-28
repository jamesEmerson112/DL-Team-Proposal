# Paper Survey for Parameter Golf

Techniques from recent papers (NeurIPS, ICLR, ICML, ACL, COLM, MLSys 2023-2025) applicable to the Parameter Golf challenge
(17M param GPT, 16 MB artifact, 10 min on 8xH100, 1024-token vocab, FineWeb).

**Last updated:** 2026-04-23
**Current SOTA:** ~1.028 BPB | **Baseline:** 1.2244 BPB | **Our best:** 1.2653 (Run 2, headwise gated attention)

---

## Summary Table

| # | Paper | Venue | Link | Key Technique | PG Leaderboard | Applicability to Parameter Golf |
|---|-------|-------|------|---------------|----------------|--------------------------------|
| 26 | **Exclusive Self-Attention (XSA)** | arXiv 2026 | [arxiv](https://arxiv.org/abs/2603.09078) | Attention orthogonal to self-value vector; forces better context modeling | **Proven** — ranks 8,10,14,15 (best 1.0979) | Consistently outperforms standard self-attention up to 2.7B. Minimal overhead. Widely adopted on leaderboard. |
| 29 | **Cross-Sequence Attention** | PG competition technique (2026) | N/A | Attend across sequence boundaries in deepest 3-4 layers during evaluation | **Proven** — ranks 14,15 (1.1307) | Eval-time only trick, no training change. Partial XSA on last 3-4 layers. High priority since validated by top submissions. |
| 15 | **Small Batch Size Training / Why Gradient Accumulation is Wasteful** | NeurIPS 2025 | [neurips](https://neurips.cc/virtual/2025/poster/119899) | Small batch sizes are stable with proper Adam hyperparameter scaling; avoiding gradient accumulation improves per-FLOP performance | Not tested | If currently using gradient accumulation, removing it and training with the natural per-GPU batch size could improve efficiency. Scale Adam beta2 to match. |
| 21 | **HybridNorm** | NeurIPS 2025 | [arxiv](https://arxiv.org/abs/2503.04598) | QKV normalization inside attention + Post-Norm in FFN; combines Pre-Norm stability with Post-Norm performance | Not tested | Cheap (just extra norm ops). Could improve training stability and final loss at 17M scale. Easy to implement: add RMSNorm to Q,K,V projections. |
| 19 | **Differential Attention (Diff Transformer)** | ICLR 2025 (Oral) | [arxiv](https://arxiv.org/abs/2410.05258) | Computes attention as the difference of two softmax maps; cancels noise, promotes sparse attention | Not tested | Lightweight (split Q/K heads, two softmax, subtract), minimal extra params. Noise-cancellation could improve BPB by helping the model attend more precisely. |
| 20 | **Value Residual Learning (ResFormer)** | ACL 2025 | [arxiv](https://arxiv.org/abs/2410.17897) | Residual connection from first layer's V matrix to all subsequent layers; mitigates attention concentration | **Mentioned** in rank 8 (1.0979) | Equivalent val loss with **16% fewer params** and **20% less data**. Nearly free: cache V from layer 0, add scaled residual to all layers. |
| 22 | **Peri-LN** | ICML 2025 | [arxiv](https://arxiv.org/abs/2502.02732) | Normalizes both input AND output of each sublayer; constrains residual spikes | Not tested | Zero extra params, just rearranging norm placement. Stabler gradients = faster convergence. Easy A/B test. |
| 1 | **Not All Tokens Are What You Need (Rho-1)** | NeurIPS 2024 Best Paper Runner-Up | [arxiv](https://arxiv.org/abs/2404.07965) | Selective Language Modeling — score tokens with a reference model, train only on high-excess-loss tokens | **Tested — FAILS at 17M scale** | Tested Option A (simple loss-threshold) at k=0.6 to k=0.95 in Session 7. Every ratio hurts BPB: k=0.6 = +0.155, k=0.8 = +0.024, k=0.95 = +0.002 vs no-SLM baseline. Paper tested at 1B+; doesn't transfer to 17M. Small models need every gradient signal. |
| 3 | **Scaling Laws with Vocabulary** | NeurIPS 2024 | [neurips](https://neurips.cc/virtual/2024/poster/93395) | Optimal vocab size scales with model size; most LLMs use too-small vocabularies | Proven — SP4096/SP8192 used in ranks 1-8 | At 17M params, 1024 tokens may actually be near-optimal or even oversized. The paper's scaling law could confirm whether 1024 is right or if 512/2048 would be better. Vocab is a free knob. |
| 4 | **Resolving Discrepancies in Compute-Optimal Scaling** | NeurIPS 2024 Spotlight | [arxiv](https://arxiv.org/abs/2406.19146) | Corrects Kaplan vs Chinchilla scaling law gap; derives optimal LR and batch size scaling laws; AdamW beta2 tuning is essential at low batch sizes | N/A (theory) | Directly useful: the paper provides formulas for optimal LR and batch size given compute budget. Also shows warmup tokens should equal model size N — so ~17M tokens warmup for a 17M model. |
| 5 | **Building on Efficient Foundations: Structured FFN Layers** | NeurIPS 2024 | [arxiv](https://arxiv.org/abs/2406.16450) | Replace dense FFN with low-rank + block-diagonal matrices; 17% throughput boost; steeper scaling curves | Not tested | Could replace the FFN with structured matrices: same model quality with 32% fewer FFN params and 1.35x training speed. Directly trades FFN params for more steps in 10 min. |
| 6 | **SwitchHead: MoE Attention** | NeurIPS 2024 | [arxiv](https://arxiv.org/abs/2312.07987) | Mixture-of-experts in attention layer; computes up to 8x fewer attention matrices; 44% compute, 27% memory | Not tested | Could dramatically speed up attention computation. At 17M params with GQA already in use, SwitchHead could stack on top for further speedup. More steps in 10 min. |
| 7 | **MoEUT: Mixture-of-Experts Universal Transformers** | NeurIPS 2024 | [neurips](https://proceedings.neurips.cc/paper_files/paper/2024/hash/321387ba926b8e58d3591c0aeb52ffc2-Abstract-Conference.html) | Layer sharing (depth recurrence) + MoE to compensate for reduced parameter count | **Proven** — depth recurrence in SOTA (rank 1, 1.0810) | Directly relevant to Parameter Golf's 16 MB cap. Share layers (fewer unique params = smaller artifact) but use MoE routing to maintain expressivity. Could cut model size while keeping quality. |
| 9 | **OneBit / BitNet** | NeurIPS 2024 / arXiv 2023 | [neurips](https://neurips.cc/virtual/2024/poster/94602) | 1-bit or 1.58-bit weight quantization during training (QAT); BitLinear as drop-in replacement for nn.Linear | Not tested | The artifact is int8+zlib compressed. Training with QAT at 1.58-bit could yield a much smaller compressed artifact (well under 16 MB), freeing budget for more parameters/layers. |
| 10 | **Compact Language Models via Pruning and Knowledge Distillation** | NeurIPS 2024 | [arxiv](https://arxiv.org/abs/2407.14679) | Structured pruning (depth, width, attention, MLP) + KD retraining with <3% of original data | Not tested | If a larger model can be trained first (say 50M), then pruned+distilled to 17M in the remaining time, the result may beat training 17M from scratch. Two-phase approach. |
| 11 | **MemoryFormer** | NeurIPS 2024 | [arxiv](https://arxiv.org/abs/2411.12992) | Replace FFN linear layers with locality-sensitive hash lookups into memory tables; near-zero FLOPs for FFN | Not tested | Radical FFN replacement. Could make each forward pass much cheaper (FFN is ~2/3 of FLOPs), allowing more steps in 10 min. The hash tables would need to fit in 16 MB though. |
| 12 | **MATES: Model-Aware Data Selection** | NeurIPS 2024 | [neurips](https://neurips.cc/virtual/2024/poster/96504) | Continuously adapt data selection to model's evolving preferences during pretraining; 2x gains over static selection, halves FLOPs to reach target loss | Not tested | Online data selection from FineWeb could double training efficiency. The influence model is tiny and runs alongside the main model. |
| 13 | **Faster LLM Training with Variable Sequence Length Curriculum** | NeurIPS 2024 | [neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/3f9bf45ea04c98ad7cb857f951f499e2-Paper-Conference.pdf) | Start with short sequences, gradually increase; reduces early-training compute waste | Not tested | Short sequences early = faster steps early = more gradient updates in 10 min. Simple to implement: just sort/bucket FineWeb by length. |
| 14 | **Surge Phenomenon in Optimal LR and Batch Size** | NeurIPS 2024 | [neurips](https://neurips.cc/virtual/2024/poster/94086) | For Adam-style optimizers, optimal LR first rises then falls as batch size increases; provides tuning guidance | N/A (theory) | Critical for 8xH100 setup where large batch sizes are natural. Helps find the right LR for the actual batch size being used. |
| 16 | **Why Warmup the Learning Rate?** | NeurIPS 2024 | [neurips](https://neurips.cc/virtual/2024/poster/95431) | Warmup forces network to well-conditioned loss landscape regions; optimal warmup tokens ~ model size N | N/A (theory) | For 17M model: warmup should be ~17M tokens. Too much warmup wastes precious steps; too little causes instability. |
| 17 | **Cross-Layer Attention (CLA)** | NeurIPS 2024 | [neurips](https://neurips.cc/virtual/2024/poster/95548) | Share KV heads between adjacent layers; 2x KV cache reduction with minimal accuracy loss | Not tested | Reduces parameter count for KV projections across layers. Combined with GQA already in use, this could further shrink the model while maintaining quality. |
| 23 | **Schedule-Free Optimizer** | NeurIPS 2024 (Oral) | [arxiv](https://arxiv.org/abs/2405.15682) | Replaces momentum with interpolation + averaging; no LR schedule needed; won MLCommons AlgoPerf | Not tested | Eliminates warmup/cooldown tuning. Perfect for 10-min wall clock where step count is unknown. Drop-in AdamW replacement. |
| 24 | **FlashAttention-3** | NeurIPS 2024 | [arxiv](https://arxiv.org/abs/2407.08608) | Warp-specialization on H100 Hopper; interleaved matmul+softmax; FP8 block quantization; 1.5-2x over FA2 | **Proven** — rank 15 (1.1307) | Direct throughput win on 8xH100 target hardware. FA3 reaches 740 TFLOP/s (75% utilization) vs FA2's 35%. Check if `train_gpt.py` already uses it. |
| 25 | **Early Weight Averaging** | COLM 2024 | [arxiv](https://arxiv.org/abs/2306.03241) | Average checkpoints along training trajectory; outperforms EMA and standard SWA | **Proven** — ranks 7,12,13,14 (EMA ~1.1228) | Already proven on PG leaderboard (~0.01 BPB drop). Free at inference time. Must-have for any competitive submission. |
| 8 | **Dynamic Layer Tying** | ICLR 2024 | [arxiv](https://arxiv.org/abs/2401.12819) | RL-based dynamic selection of which layers to tie during training; 75-87% parameter reduction | Not tested | Extreme parameter reduction for the 16 MB constraint. If 9 layers can be collapsed to 2-3 unique layers + routing, the artifact shrinks dramatically while maintaining perplexity. |
| 27 | **Mixture-of-Depths (MoD)** | arXiv 2024 (DeepMind) | [arxiv](https://arxiv.org/abs/2404.02258) | Learned router decides per-token whether to run full block or skip; top-k routing | Not tested | Up to 50% FLOPs savings = faster steps. But router adds params and complexity at 17M scale. Worth prototyping if throughput is the bottleneck. |
| 28 | **Gated Linear Attention (GLA)** | ICML 2024 | [arxiv](https://arxiv.org/abs/2312.06635) | Linear attention with data-dependent gates; O(n) complexity; FlashLinearAttention kernel | Not tested | Risky: PG uses 1024-token seqs where quadratic attention is already fast. O(n) benefit shows at longer contexts. Lower priority. |
| 2 | **Scaling Data-Constrained Language Models** | NeurIPS 2023 Outstanding Paper | [arxiv](https://arxiv.org/abs/2305.16264) | Scaling laws for repeated data — up to 4 epochs of repeated data has negligible loss vs unique data | N/A (theory) | Validates that if the 10-min window limits how much FineWeb you see, repeating data 2-4x is fine. Train longer on same data without penalty. |
| 18 | **Are Emergent Abilities a Mirage?** | NeurIPS 2023 Outstanding Paper | [arxiv](https://arxiv.org/abs/2304.15004) | Apparent emergence is an artifact of discontinuous metrics, not model behavior | N/A (theory) | Not directly actionable for BPB optimization, but reinforces that BPB (a continuous metric) is the right evaluation target — small models improve smoothly, no "emergence threshold" to worry about. |
| 30 | **ShishuLM: Low Attention Transformer Models** | arXiv 2025 | [arxiv](https://arxiv.org/abs/2510.13860) | Asymmetric blocks: early layers need attention (token mixing), late layers need MLP (prediction refinement); MLP-only layers with shared weights | Not tested | Explores whether attention and MLP layers can have different counts. At PG scale, confirms depth recurrence intuition: the middle transition layers (4-5) are where repeated compute helps most. Weight-shared MLP-only layers could compress further. |
| 31 | **Reducing the Transformer Architecture to a Minimum** | arXiv 2024 | [arxiv](https://arxiv.org/abs/2410.13732) | Systematically removes transformer components to find minimal viable architecture | Not tested | Useful reference for understanding which components are essential vs redundant at small scale. Could inform aggressive simplification strategies under 16 MB budget. |
| 32 | **Attention-Only Transformers** | arXiv 2023 | [arxiv](https://arxiv.org/abs/2309.08593) | Proves MLPs can be replaced by attention heads; converts MLP-and-attention transformer into attention-only at the cost of more heads | N/A (theory) | Theoretical result showing MLP and attention are interchangeable. Not practical for PG (requires many more heads = more params), but confirms that the attention/MLP boundary is softer than it appears. |

---

## Leaderboard-Proven Techniques

Techniques confirmed used by top PG submissions as of April 2026:

| Technique | Paper # | Best Leaderboard BPB | Submissions | Type |
|-----------|---------|---------------------|-------------|------|
| EMA / Weight Averaging | #25 | 1.1228 | Ranks 7, 12, 13, 14 | Post-training |
| XSA / Cross-Sequence Attention | #26, #29 | 1.0979 | Ranks 8, 10, 14, 15 | Eval-time |
| FlashAttention-3 | #24 | 1.1307 | Rank 15 | Throughput |
| Value Residual Learning | #20 | 1.0979 (mentioned) | Rank 8 | Architecture |
| Depth Recurrence (MoEUT-style) | #7 | 1.0810 (SOTA) | Rank 1 | Architecture |
| Vocab Scaling (SP4096/SP8192) | #3 | 1.0810 (SOTA) | Ranks 1-8 | Data/Tokenizer |
| Gated Attention (our runs) | N/A | 1.2653 (our Run 2) | Our experiment | Architecture |

Additional confirmed leaderboard techniques (not from papers above):
- **LoRA TTT** — test-time training with low-rank updates
- **Custom tokenizers** — SP8192, CaseOps
- **LoRA on tied embeddings**
- **Pre-quantization TTT**
- **GPTQ Embeddings + SDClip** — Hessian-aware quantization with per-layer clipping
- **MuonEq-R** — row-normalized Muon optimizer
- **Parallel Residuals** — GPT-J style separate attention/MLP residual streams
- **QK-Gain** — learnable per-head attention scaling (optimal 5.25)

Current best: **~1.028 BPB** (down from 1.2244 baseline).

---

## Top 8 Most Actionable Techniques (Ranked by Impact/Effort)

### 1. Value Residual Learning (ResFormer) — Paper #20
**Impact: HIGH | Effort: LOW | Leaderboard: Mentioned in rank 8**
Cache V from layer 0, add a scaled residual V_0 to each subsequent layer's V matrix. Achieves equivalent val loss with 16% fewer parameters and 20% less training data. At 17M scale, that's like a free ~2.7M parameter boost in effective capacity. Also reduces attention sinks which hurt BPB. Simple to implement: one cached tensor + one addition per layer.

### ~~2. Rho-1 Selective Token Training — Paper #1~~ DOES NOT WORK
**Impact: ~~HIGH~~ NEGATIVE | Effort: MEDIUM | Leaderboard: Tested by us — fails at 17M scale**
~~Train a small reference model first (or use the model's own early checkpoint), score FineWeb tokens by excess loss, mask out low-value tokens from the loss computation.~~ **Session 7 validated:** Option A (simple loss-threshold, keep top-k% by raw loss) tested at k=0.6, 0.7, 0.8, 0.95 on both SP1024 and SP8192 configs. Every ratio hurts BPB — k=0.6 adds +0.155, k=0.95 adds +0.002. The trend is monotonic: less filtering = less damage, optimal k = 1.0 (no filtering). At 17M params, the model needs every token's gradient. Paper results (1B+ models) do not transfer down. **Moved to "Techniques That Didn't Work" in findings.md.**

### 3. Early Weight Averaging / EMA — Paper #25
**Impact: PROVEN ~0.01 BPB | Effort: LOW | Leaderboard: Proven (ranks 7,12,13,14)**
Average checkpoints along the training trajectory. Free at inference time — just average weights post-training. Must-have for any competitive submission. Save checkpoints every N steps in the last phase of training, average them. Already validated by multiple leaderboard entries.

### 4. Peri-LN or HybridNorm — Papers #21, #22
**Impact: MEDIUM | Effort: LOW | Leaderboard: Not tested**
Better normalization placement = stabler gradients = faster convergence. Zero extra parameters. Peri-LN normalizes both input AND output of each sublayer. HybridNorm adds QKV norm + Post-Norm FFN. Easy A/B test with minimal code changes.

### 5. Differential Attention — Paper #19
**Impact: HIGH potential | Effort: MEDIUM | Leaderboard: Not tested**
Compute attention as the difference of two softmax maps. ICLR 2025 Oral. Noise-cancellation promotes sparse, precise attention. Lightweight modification (split Q/K, two softmax, subtract). At 17M scale the per-head overhead is tiny.

### 6. Schedule-Free Optimizer — Paper #23
**Impact: MEDIUM | Effort: LOW | Leaderboard: Not tested**
No LR schedule needed — perfect for fixed 10-min wall clock where step count is unknown. Won MLCommons 2024 AlgoPerf Self-Tuning track. Drop-in AdamW replacement. Could combine with Muon for matrix params and Schedule-Free Adam for embeddings/scalars.

### 7. Structured FFN (Low-Rank + Block-Diagonal) — Paper #5
**Impact: HIGH | Effort: MEDIUM | Leaderboard: Not tested**
Replace the dense FFN with structured matrices. This gives 1.35x training speedup (more steps in 10 min) AND steeper scaling curves (better loss per FLOP). The FFN is typically 2/3 of transformer parameters — making it structured could allow a wider or deeper model within the same 16 MB budget.

### 8. Layer Tying / Depth Recurrence (MoEUT style) — Paper #7
**Impact: HIGH | Effort: HIGH | Leaderboard: Proven (SOTA rank 1, 1.0810)**
Share weights across layers but add lightweight per-layer routing or adapters. A 9-layer model with 3 unique layer groups would have ~1/3 the parameters but similar effective depth. The freed parameter budget goes to wider dimensions or more heads. This is the single biggest lever for the 16 MB constraint. Already in SOTA submission.

---

## Techniques Already in Use (Confirmed)

- **GQA (Grouped Query Attention)** — already reduces KV parameters
- **RoPE** — standard positional encoding
- **RMSNorm** — efficient normalization
- **ReLU^2** — activation function
- **Muon optimizer** — second-order optimizer, already ~1.4x faster than AdamW
- **int8 + zlib compression** — for the 16 MB artifact constraint
- **Gated Attention (headwise)** — NeurIPS 2025 Best Paper, +37K params, our Run 2 (1.2653)

## Techniques to Investigate Further

- **Differential Attention (#19)**: Two-softmax-subtract attention, ICLR 2025 Oral. Could stack with gated attention.
- **Peri-LN (#22) or HybridNorm (#21)**: Better norm placement, zero params. Easy A/B test.
- **Schedule-Free Optimizer (#23)**: Eliminates LR schedule tuning. NeurIPS 2024 Oral.
- **BitNet / QAT at 1.58-bit**: Could the model be trained at lower precision, yielding a smaller compressed artifact and freeing parameter budget?
- **Cross-Layer Attention + GQA**: Stacking CLA on top of GQA for maximum KV parameter sharing
- **MemoryFormer hash tables**: Radical but could be transformative if hash tables compress well under zlib
- **Muon -> NorMuon upgrade**: 11% improvement over Muon with same memory footprint (NeurIPS 2025 workshop paper)
- **Exclusive Self-Attention (#26)**: Orthogonal attention, proven on leaderboard (ranks 8,10,14,15)

---

## Papers Blocked or Low Priority for PG

| Paper | Why Skip |
|-------|----------|
| MiniPLM (ICLR 2025) — KD for pretraining | Requires offline teacher inference before 10-min clock; likely violates PG rules |
| Gated Linear Attention (ICML 2024) | PG uses 1024-token seqs; O(n) benefit only at longer contexts. Large architectural change, high risk |
| Mixture-of-Depths (DeepMind 2024) | Router adds params at 17M scale; FLOPs savings vs parameter cost tradeoff unclear |

---

## Sources

### NeurIPS 2023-2024 Papers
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
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Early Weight Averaging meets High Learning Rates](https://arxiv.org/abs/2306.03241)

### ICLR / ICML / ACL / Other 2024-2025 Papers
- [Differential Attention (Diff Transformer)](https://arxiv.org/abs/2410.05258)
- [Value Residual Learning (ResFormer)](https://arxiv.org/abs/2410.17897)
- [HybridNorm](https://arxiv.org/abs/2503.04598)
- [Peri-LN](https://arxiv.org/abs/2502.02732)
- [Schedule-Free Optimizer (The Road Less Scheduled)](https://arxiv.org/abs/2405.15682) | [Code](https://github.com/facebookresearch/schedule_free)
- [Gated Linear Attention](https://arxiv.org/abs/2312.06635)
- [Exclusive Self-Attention](https://arxiv.org/abs/2603.09078)
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)
- [MiniPLM](https://arxiv.org/abs/2410.17215)

### Asymmetric / Minimal Architecture Papers
- [ShishuLM: Low Attention Transformer Models](https://arxiv.org/abs/2510.13860)
- [Reducing the Transformer Architecture to a Minimum](https://arxiv.org/abs/2410.13732)
- [Attention-Only Transformers and Implementing MLPs with Attention Heads](https://arxiv.org/abs/2309.08593)

### Other References
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
- [NorMuon](https://arxiv.org/abs/2510.05491)
