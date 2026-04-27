# Paper Summaries for Team — Related Work

**Author:** James Vo
**Date:** 2026-04-22
**Purpose:** Quick-reference summaries so teammates can understand our Related Work citations without reading the full papers.

## Quick Reference

| # | Paper | Year | Priority | Why It Matters |
|---|-------|------|----------|----------------|
| 1 | FineWeb Datasets (Penedo et al.) | 2024 | LOW | Our training data comes from this |
| 2 | Chinchilla Scaling Laws (Hoffmann et al.) | 2022 | MEDIUM | Justifies our small-model-more-data strategy |
| 3 | NanoGPT Speedrun Benchmark (Zhao et al.) | 2025 | MEDIUM | Formalizes the speedrun tradition our codebase descends from |
| 4 | FlashInfer (Ye et al.) | 2025 | LOW | Systems/inference paper, tangential to our training focus |

---

## Paper 1: FineWeb — Decanting Web Data at Scale

**Authors:** Guilherme Penedo, Hynek Kydlicek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf (HuggingFace, 2024)

**Summary:** FineWeb is a 15-trillion-token dataset derived from 96 Common Crawl snapshots. The authors developed a multi-stage pipeline — URL filtering, language detection, MinHash deduplication, and quality heuristics — that outperforms prior open datasets (C4, Dolma, RedPajama, RefinedWeb) on standard LLM benchmarks. The standout contribution is **FineWeb-Edu**, a 1.3T-token educational subset selected by a classifier trained on LLM-annotated quality scores. FineWeb-Edu dramatically improves knowledge-intensive benchmarks (MMLU, ARC) even at small model scales.

**Why it matters for us:** Our training data (`fineweb_sp1024`, the 1024-vocab tokenization of FineWeb) comes directly from this work. Data quality is **fixed** for Parameter Golf — we can't swap datasets or re-tokenize. But understanding the filtering pipeline matters for our Related Work section: it explains why our data quality is high and why we shouldn't expect gains from data-side ablations.

**Priority: LOW** — Read the abstract and Section 2 (pipeline overview). Skip the detailed ablation tables unless you're writing the data section of our paper.

---

## Paper 2: Training Compute-Optimal Large Language Models (Chinchilla)

**Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (DeepMind, 2022)

**Summary:** DeepMind trained over 400 models (70M to 16B parameters) varying both model size and training tokens to find compute-optimal scaling laws. Their key finding: **model size and training data should scale proportionally** — previous models (GPT-3, Gopher) were too large for their training budgets. To prove it, they trained Chinchilla (70B params) on 4x more data than the larger Gopher (280B), and Chinchilla won on nearly every benchmark. The paper establishes that for a fixed compute budget, you're often better off training a smaller model on more tokens.

**Why it matters for us:** This directly justifies our Parameter Golf strategy. Under a 10-minute wall clock and 16 MB model size cap, we can't go big on parameters. Chinchilla tells us that's fine — **push for more training tokens, not a bigger model.** Our 17M-parameter architecture is small, but if we maximize throughput (more steps in 10 minutes via 8 GPUs), the scaling laws predict we'll keep improving. This is also why we care about training speed optimizations (batch size, learning rate schedules) more than architectural size.

**Priority: MEDIUM** — Read the abstract and Figure 1 (the scaling curves). The core insight ("match model size to data budget") is all you need for our experiments. Skip the appendix with the 400-model grid.

---

## Paper 3: NanoGPT Speedrun Benchmark

**Authors:** Xihui Zhao, Zitong Yang, Jiaxun Li, Himanshu Tyagi, Colin Wei (2025)

**Summary:** This paper wraps 19 of the modded-nanogpt community speedrun records into a formal AI agent benchmark. Each task asks an agent to reproduce a known training speedup — things like switching to Muon optimizer, adding U-Net skip connections, or tuning learning rate schedules. The key finding: **even frontier LLMs struggle to reproduce improvements that humans discovered through iterative experimentation.** The benchmark reveals that "knowing about" a technique (from papers or documentation) is very different from successfully implementing it in a specific codebase.

**Why it matters for us:** The speedrun records are our codebase's direct ancestry — Parameter Golf is a fork of modded-nanogpt (see note 14 for the full lineage). Our ablation approach — isolating techniques and measuring their individual BPB deltas — mirrors what this paper formalizes as benchmark tasks. Good citation for our Related Work to establish that systematic technique evaluation in this codebase family is an active area of research, not just hobbyist benchmarking.

**Priority: MEDIUM** — Good for Related Work citation and framing. Read the intro and task list (Table 1). You don't need to read the agent evaluation results unless you're curious about LLM coding capabilities.

---

## Paper 4: FlashInfer — Efficient and Customizable Attention Engine for LLM Inference Serving

**Authors:** Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen (University of Washington, CMU, OctoAI, 2025)

**Summary:** FlashInfer is a system for accelerating LLM inference serving through two innovations: (1) **block-sparse attention** that represents KV-caches as block-sparse matrices, enabling shared-prefix and disaggregated serving patterns; and (2) a **JIT-compiled composable attention kernel generator** that fuses custom attention variants (RoPE, ALiBi, sliding window, quantized KV) into single GPU kernels. The system achieves 29-69% inter-token latency reduction compared to existing solutions like vLLM's paged attention.

**Why it matters for us:** Honestly — **it doesn't much.** FlashInfer is an inference serving optimization (making already-trained models serve requests faster). Our project is about training-time optimization under the Parameter Golf constraints. We won't be running inference servers. I included it in the reading list because it's a strong systems paper that came up during the literature survey, but it's only relevant if our paper's Related Work section touches on the broader LLM systems landscape or if we discuss what happens after training.

**Priority: LOW** — Skip unless we decide to include an inference/deployment section in the paper. If we do, the abstract alone has what we need.

---

## How These Papers Connect

```
Chinchilla (2022)                     FineWeb (2024)
"scale data, not just params"         "15T tokens, cleaned right"
         │                                   │
         │  justifies our                    │  provides our
         │  small-model strategy             │  training data
         │                                   │
         ▼                                   ▼
    ┌─────────── Parameter Golf ───────────────┐
    │  17M params, 10 min, 16 MB, 8×H100      │
    │  Goal: lowest val_bpb                    │
    └───────────────┬──────────────────────────┘
                    │
                    │  descended from
                    ▼
         NanoGPT Speedrun Benchmark (2025)
         "formalizes modded-nanogpt records
          as systematic benchmarks"

    FlashInfer (2025) ──── tangential ────
    "inference serving, not training"
```

## Reading Order (if you have limited time)

1. **Chinchilla** abstract + Figure 1 (~5 min) — most relevant to our strategy
2. **NanoGPT Speedrun** intro + Table 1 (~10 min) — context for our codebase lineage
3. **FineWeb** abstract + Section 2 (~5 min) — context for our data
4. **FlashInfer** — skip unless writing inference-related sections
