# Paper Survey: Techniques for Parameter Golf

> Searched 2026-04-22. Focus: recent papers (2024-2025) with techniques applicable to
> ~17M param GPT, 8xH100 for 10 min, 16 MB compressed, minimizing val BPB on FineWeb.
> Current SOTA: ~1.028 BPB. Baseline: 1.2244 BPB.

---

## 1. Differential Attention (DIFF TRANSFORMER)

| Field | Value |
|-------|-------|
| Title | Differential Transformer |
| Authors | Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei |
| Venue | **ICLR 2025 (Oral)** |
| arXiv | https://arxiv.org/abs/2410.05258 |
| Technique | Computes attention as the *difference* of two softmax maps. Partitions Q,K into two groups, takes two separate softmax attention maps, subtracts one from the other. Cancels noise, promotes sparse attention. |

**PG Assessment:** Strong candidate. The modification is lightweight (split Q/K heads, two softmax, subtract) and adds minimal parameters. At 17M scale the per-head overhead is tiny. The noise-cancellation property could improve BPB by helping the model attend more precisely. Straightforward to implement in `train_gpt.py` by modifying the attention function.

---

## 2. Mixture-of-Depths (MoD)

| Field | Value |
|-------|-------|
| Title | Mixture-of-Depths: Dynamically allocating compute in transformer-based language models |
| Authors | David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro |
| Venue | arXiv 2024 (Google DeepMind) |
| arXiv | https://arxiv.org/abs/2404.02258 |
| Technique | Uses a learned router to decide per-token whether to run the full transformer block (self-attention + MLP) or skip via residual connection. Top-k routing keeps compute graph static. |

**PG Assessment:** Mixed value for PG. The main benefit is FLOPs savings (up to 50% fewer per forward pass) which translates to faster steps, letting you train more steps in 10 min. However, the router adds parameters and complexity, and at 17M scale every parameter counts. The static top-k routing is hardware-friendly. Worth prototyping if throughput is the bottleneck rather than model capacity.

---

## 3. Schedule-Free Optimizer

| Field | Value |
|-------|-------|
| Title | The Road Less Scheduled |
| Authors | Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky |
| Venue | **NeurIPS 2024 (Oral)** |
| arXiv | https://arxiv.org/abs/2405.15682 |
| Code | https://github.com/facebookresearch/schedule_free |
| Technique | Replaces momentum with a combination of interpolation and averaging. No learning rate schedule needed. Won MLCommons 2024 AlgoPerf Self-Tuning track. |

**PG Assessment:** High potential. Eliminates the need to tune warmup/cooldown steps, which is critical when you have exactly 10 min and don't know how many steps you'll get. The schedule-free property means the optimizer works well regardless of when training stops. Drop-in replacement for AdamW. Could combine with Muon for matrix params and Schedule-Free Adam for embeddings/scalars.

---

## 4. HybridNorm

| Field | Value |
|-------|-------|
| Title | HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization |
| Authors | Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, Jinwen Ma |
| Venue | **NeurIPS 2025** |
| arXiv | https://arxiv.org/abs/2503.04598 |
| Technique | QKV normalization inside attention + Post-Norm in FFN. Combines Pre-Norm stability with Post-Norm performance. |

**PG Assessment:** Promising and cheap. Adding QKV norm is just a few extra norm operations (negligible parameters, small compute). Could improve training stability and final loss at 17M scale, where training dynamics matter more per-step. Easy to implement: add RMSNorm to Q, K, V projections and switch FFN sublayer to post-norm.

---

## 5. Peri-LN (Peripheral Layer Normalization)

| Field | Value |
|-------|-------|
| Title | Peri-LN: Revisiting Normalization Layer in the Transformer Architecture |
| Authors | Jeonghoon Kim, Byeongchan Lee, Cheonbok Park, Yeontaek Oh, Beomjun Kim, Taehwan Yoo, Seongjin Shin, Dongyoon Han, Jinwoo Shin, Kang Min Yoo |
| Venue | **ICML 2025** |
| arXiv | https://arxiv.org/abs/2502.02732 |
| Technique | Normalizes both input AND output of each sublayer. Constrains residual spikes from Pre-LN while maintaining stronger gradient flow than Post-LN. |

**PG Assessment:** Low-risk improvement. Zero extra parameters, just rearranging where LayerNorm/RMSNorm is applied. Reports more balanced variance growth and steadier gradient flow. At 17M scale with short training, stable gradients = faster convergence to lower loss. Easy A/B test.

---

## 6. Value Residual Learning (ResFormer)

| Field | Value |
|-------|-------|
| Title | Value Residual Learning For Alleviating Attention Concentration In Transformers |
| Authors | Zhongzhi Yu et al. |
| Venue | **ACL 2025** |
| arXiv | https://arxiv.org/abs/2410.17897 |
| Technique | Adds residual connection from first layer's value matrix to all subsequent layers. Mitigates attention concentration (deeper layers over-focusing on few tokens). |

**PG Assessment:** Very strong candidate for PG. Achieves equivalent val loss with **16% fewer parameters** and **20% less training data**. At 17M scale, that's like getting a free ~2.7M parameter boost in effective capacity. Implementation is simple: cache V from layer 0, add scaled residual V_0 to each subsequent layer's V. Minimal compute overhead. Also reduces attention sinks which hurt BPB.

---

## 7. FlashAttention-3

| Field | Value |
|-------|-------|
| Title | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| Authors | Tri Dao et al. |
| Venue | **NeurIPS 2024** |
| arXiv | https://arxiv.org/abs/2407.08608 |
| Technique | Exploits H100 Hopper architecture: warp-specialization for overlapping compute/data movement, interleaved matmul+softmax, FP8 block quantization. 1.5-2x speedup over FA2. |

**PG Assessment:** Direct throughput win. PG runs on 8xH100 -- exactly the target hardware. FA3 reaches 740 TFLOP/s (75% utilization) vs FA2's 35% utilization on H100. More steps in 10 min = lower final BPB. Check if `train_gpt.py` already uses FA3 via PyTorch's `scaled_dot_product_attention`; if not, switching is a one-line change that could nearly double attention throughput.

---

## 8. Gated Linear Attention (GLA)

| Field | Value |
|-------|-------|
| Title | Gated Linear Attention Transformers with Hardware-Efficient Training |
| Authors | Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim |
| Venue | **ICML 2024** |
| arXiv | https://arxiv.org/abs/2312.06635 |
| Technique | Linear attention variant with data-dependent gates. FlashLinearAttention kernel is faster than FlashAttention-2 even on short sequences. O(n) complexity. |

**PG Assessment:** Interesting but risky. Faster than FA2 on short seqs, but PG uses 1024-token sequences where standard attention is already efficient. The O(n) benefit shows at longer contexts. At 17M scale, the architectural change is large and could hurt BPB vs well-tuned standard attention. Lower priority unless sequence length is increased substantially.

---

## 9. Early Weight Averaging

| Field | Value |
|-------|-------|
| Title | Early Weight Averaging meets High Learning Rates for LLM Pre-training |
| Authors | Sanyal et al. |
| Venue | **COLM 2024** |
| arXiv | https://arxiv.org/abs/2306.03241 |
| Technique | Average checkpoints along the training trajectory. Works best with high learning rates and substantial spacing between averaged checkpoints. Outperforms EMA and standard SWA. |

**PG Assessment:** Already proven in PG competition -- model averaging is confirmed to have dropped BPB from ~1.20 to ~1.19 in early submissions. The technique is free at inference time (just average weights post-training). Can be combined with any architecture. Must-have for any competitive submission. Implementation: save checkpoints every N steps in the last phase of training, average them.

---

## 10. MiniPLM (KD for Pretraining)

| Field | Value |
|-------|-------|
| Title | MiniPLM: Knowledge Distillation for Pre-Training Language Models |
| Authors | Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang |
| Venue | **ICLR 2025** |
| arXiv | https://arxiv.org/abs/2410.17215 |
| Technique | Offline teacher inference to refine training data distribution. Teacher guides student pretraining without adding training-time compute. Works across model families. |

**PG Assessment:** Not applicable to PG as structured. PG requires training from scratch in 10 min -- there's no time budget for teacher inference, and no pre-existing teacher model. The offline data-refinement step would need to happen before the 10-min clock starts, which may violate rules. Skip for PG, but relevant for the CS 7643 project.

---

## 11. Exclusive Self-Attention (XSA)

| Field | Value |
|-------|-------|
| Title | Exclusive Self Attention |
| Authors | (arXiv March 2026) |
| Venue | arXiv preprint |
| arXiv | https://arxiv.org/abs/2603.09078 |
| Technique | Constrains attention to capture only information orthogonal to the token's own value vector, excluding self-position information. Forces better context modeling. |

**PG Assessment:** Promising. Consistently outperforms standard self-attention across model sizes up to 2.7B, with gains increasing at longer sequences. Minimal computational overhead. The orthogonality constraint is elegant and cheap to compute. Worth testing at 17M scale, though the gains may be smaller at short (1024-token) sequences.

---

## 12. Cross-Sequence Attention (from PG leaderboard)

| Field | Value |
|-------|-------|
| Title | (Competition technique, not a paper) |
| Source | Parameter Golf leaderboard submissions |
| Technique | Apply attention across sequence boundaries in the deepest 3-4 layers, borrowing context from adjacent sequences during evaluation. |

**PG Assessment:** Already proven on the PG leaderboard. Partial XSA (last 3-4 layers only) keeps overhead manageable. This is an evaluation-time trick that improves BPB without changing training. High priority to implement since it's been validated by top submissions.

---

## Priority Ranking for Parameter Golf

Sorted by expected BPB improvement per implementation effort:

| Rank | Technique | Effort | Expected Impact | Risk |
|------|-----------|--------|-----------------|------|
| 1 | **Early Weight Averaging** | Low | Proven ~0.01 BPB | Very low |
| 2 | **Value Residual Learning** | Low | 16% param-equivalent boost | Low |
| 3 | **Cross-Sequence Attention** | Medium | Proven on leaderboard | Low |
| 4 | **Differential Attention** | Medium | Better attention precision | Medium |
| 5 | **Peri-LN / HybridNorm** | Low | Stabler training, faster convergence | Very low |
| 6 | **FlashAttention-3** | Low | More steps in 10 min | Very low |
| 7 | **Schedule-Free Optimizer** | Low | Better anytime stopping | Low |
| 8 | **XSA (Exclusive Self-Attn)** | Medium | Consistent small gains | Medium |
| 9 | **Mixture-of-Depths** | High | More steps via FLOPs savings | Medium |
| 10 | **GLA** | High | Speed on short seqs | High |
| 11 | **MiniPLM** | N/A | Requires teacher | Blocked by rules |

---

## Techniques Already Proven on PG Leaderboard (from search)

These are confirmed used by top submissions as of April 2026:
- **Model/weight averaging** (checkpoint averaging post-training)
- **Cross-sequence attention** (partial, last 3-4 layers)
- **LoRA TTT** (test-time training with low-rank updates)
- **Custom tokenizers** (SP8192, CaseOps)
- **LoRA on tied embeddings**
- **Pre-quantization TTT**
- Current best: ~1.028 BPB (down from 1.2244 baseline)
