# NeurIPS Papers — Related Work Reference

> 6 papers relevant to our project (nanochat training pipeline + Parameter Golf competition).

## Paper 1: Gated Attention for Large Language Models

**NeurIPS 2025 — Best Paper Award (Main Track)**

- **Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin (Alibaba Qwen)
- **Paper:** [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) | [OpenReview](https://openreview.net/forum?id=1b7whO4SfY) | [GitHub](https://github.com/qiuzh20/gated_attention)

### Summary

Applies head-specific sigmoid gating after scaled dot-product attention in transformers. Tested on 15B mixture-of-experts and 1.7B dense models trained on 3.5 trillion tokens. Shows consistent improvement in training stability, eliminates attention sink phenomena, and improves long-context extrapolation.

### Key Findings

- Adding a simple gate (sigmoid) per attention head improves performance across 30+ experiments
- Mitigates "attention sinks" — the phenomenon where models waste attention on the first token
- Improves long-context extrapolation without additional training

### Relevance to Our Project

- **nanochat** uses Multi-Query Attention (MQA) with QK normalization — this paper suggests gating could further improve it
- **Parameter Golf** — attention efficiency matters when you only have 16 MB and 10 minutes
- Potential experiment: add gated attention to nanochat's architecture and measure impact on learning curves

---

## Paper 2: Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

**NeurIPS 2025 — Runner-Up (Main Track)**

- **Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang (Tsinghua University)
- **Paper:** [arXiv:2504.13837](https://arxiv.org/abs/2504.13837) | [OpenReview](https://openreview.net/forum?id=4OsgYD7em5) | [Project Page](https://limit-of-rlvr.github.io/)

### Summary

Tests whether Reinforcement Learning with Verifiable Rewards (RLVR) gives LLMs genuinely new reasoning abilities. Finding: RL improves sampling efficiency but doesn't develop fundamentally new reasoning patterns — capabilities remain bounded by what the base model already has.

### Key Findings

- RLVR makes the model more likely to produce correct answers on the first try
- But when given many samples (e.g., best-of-N), base models match or outperform RL-trained ones
- RL redistributes probability mass toward correct answers, rather than creating new reasoning paths

### Relevance to Our Project

- **nanochat Stage 4 (RL/GRPO)** is optional — this paper provides evidence for why it might have limited value
- Experiment idea: compare model performance with and without Stage 4, measuring both single-sample and best-of-N accuracy
- Useful for the final paper's Discussion section — "is RL worth the compute?"

---

## Paper 3: Superposition Yields Robust Neural Scaling

**NeurIPS 2025 — Runner-Up (Main Track)**

- **Authors:** Yizhou Liu, Ziming Liu, Jeff Gore (MIT)
- **Paper:** [arXiv:2505.10465](https://arxiv.org/abs/2505.10465) | [OpenReview](https://openreview.net/forum?id=knPz7gtjPW) | [GitHub](https://github.com/liuyz0/SuperpositionScaling)

### Summary

Proposes that neural scaling laws (Chinchilla scaling) arise from representation superposition — models pack more features into hidden dimensions than they can cleanly separate. Under strong superposition, loss scales inversely with model dimension, matching observed Chinchilla scaling behavior.

### Key Findings

- Scaling laws aren't mysterious — they emerge naturally from how models compress features
- Loss ∝ 1/dimension under superposition, consistent with Chinchilla scaling
- Explains why larger models are more sample-efficient

### Relevance to Our Project

- **nanochat's `--depth` parameter** is built directly on Chinchilla-optimal scaling laws (20:1 token-to-parameter ratio)
- This paper explains *why* that scaling works
- **Parameter Golf** — understanding superposition could help design better architectures under the 16 MB constraint
- Useful for the proposal's Related Work section

---

## Paper 4: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

**NeurIPS 2024 — Spotlight (Datasets & Benchmarks Track)**

- **Authors:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (Hugging Face)
- **Link:** [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html)

### Summary

Introduces FineWeb — a 15-trillion token dataset of curated web text from 96 Common Crawl snapshots. Also introduces FineWeb-EDU, a filtered subset focused on educational content. Details the filtering pipeline: URL dedup, MinHash dedup, quality filtering, and educational content scoring.

### Key Findings

- FineWeb outperforms other open pretraining datasets (C4, Dolma, RedPajama, RefinedWeb)
- FineWeb-EDU (educational subset) produces even better models despite being smaller
- Quality filtering matters more than dataset size
- Sufficient data to train a Chinchilla-optimal 500B+ parameter model

### Relevance to Our Project

- **This is our dataset.** nanochat pretrains on FineWeb-EDU; Parameter Golf evaluates on FineWeb validation
- Understanding the filtering pipeline helps interpret model behavior
- Data quality insights could inform Parameter Golf strategy (which shards are most valuable?)
- Essential citation for the proposal's Datasets section

---

## Paper 5: DataComp-LM: In Search of the Next Generation of Training Data

**NeurIPS 2024 — Datasets & Benchmarks Track**

- **Authors:** Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. (DCLM consortium)
- **Paper:** [arXiv:2406.11794](https://arxiv.org/abs/2406.11794) | [OpenReview](https://openreview.net/forum?id=CNWdWn47IE) | [GitHub](https://github.com/mlfoundations/dclm)

### Summary

Introduces DataComp-LM (DCLM) — a benchmark and testbed for developing better training datasets for LLMs. Defines the DCLM CORE score (22-task evaluation ensemble) and the DCLM-BASELINE dataset. Shows that data curation alone can match or beat models trained on 10x more data.

### Key Findings

- Data quality outweighs data quantity — curated 2T tokens beats raw 10T tokens
- DCLM-BASELINE achieves state-of-the-art compute-performance tradeoffs
- The 22-task CORE benchmark covers: reasoning (ARC), knowledge (MMLU, TriviaQA), math (GSM8K), code (HumanEval), common sense (HellaSwag), and more

### Relevance to Our Project

- **nanochat evaluates on DCLM CORE** — this paper defines what that benchmark measures and why
- Understanding CORE helps interpret nanochat's benchmark scores (e.g., CORE 0.22 for depth-20)
- Data curation insights apply to both nanochat (choosing pretraining data) and Parameter Golf (optimizing within 10 min)
- Essential citation for the proposal's Resources/Related Work section

---

## Paper 6: FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

**MLSys 2025 — Best Paper Award**

- **Authors:** Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Baris Kasikci, Arvind Krishnamurthy, Luis Ceze, Vinod Grover, Tianqi Chen (UW, NVIDIA, CMU, Perplexity, SGLang)
- **Paper:** [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) | [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf) | [GitHub](https://github.com/flashinfer-ai/flashinfer)

### Summary

Introduces FlashInfer — a customizable attention engine for LLM inference serving. Addresses KV-cache storage heterogeneity with a unified block-sparse row (BSR) format, provides JIT-compiled attention templates for custom kernels, and uses load-balanced dynamic scheduling compatible with CUDAGraph for diverse serving workloads.

### Key Findings

- Unified BSR format represents all KV-cache layouts (paged, radix-tree, tree-masked) under one kernel family
- JIT-compiled templates let users create custom fused kernels (e.g., RoPE-fused attention) in ~20 lines of code
- 29–69% inter-token latency reduction vs. compiler backends; 28–30% latency reduction for long-context inference
- 70–83% bandwidth utilization on skewed batches vs. ~45% for FlashAttention
- Adopted by SGLang, vLLM, and MLC Engine; NVIDIA releasing TensorRT-LLM kernels through FlashInfer

### Relevance to Our Project

- **nanochat uses Flash Attention 3** with SDPA fallback — FlashInfer extends and generalizes Flash Attention for serving workloads
- **nanochat's KV cache** (`engine.py`) could benefit from FlashInfer's unified BSR format for more efficient inference
- **Parameter Golf** — JIT-fused custom kernels could reduce latency under tight compute constraints
- Systems-level complement to the architecture/theory papers above

---

## Quick Reference

| # | Paper | Year | Track | Type |
|---|---|---|---|---|
| 1 | Gated Attention for LLMs | NeurIPS 2025 | Main | Architecture |
| 2 | Does RL Really Incentivize Reasoning? | NeurIPS 2025 | Main | Training (RL) |
| 3 | Superposition Yields Robust Neural Scaling | NeurIPS 2025 | Main | Theory (scaling) |
| 4 | The FineWeb Datasets | NeurIPS 2024 | Datasets | **Dataset** |
| 5 | DataComp-LM (DCLM) | NeurIPS 2024 | Datasets | Benchmark |
| 6 | FlashInfer | MLSys 2025 | Systems | **Inference** |
