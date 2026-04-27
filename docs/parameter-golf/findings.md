# Parameter Golf — Findings & Insights

> What we learned from participating. Insights that could feed back into the course project.

## Official Runs — 8×H100 (sorted by BPB, best first)

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Size (int8+zlib) | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------------------|---------|
| 10 | SP8192 combo + TTT | 20.77M | 3.0666 | 1.1872 | 10,582 | 57ms | 19.41 MB | **No** |
| 11 | **SP8192 combo slim + TTT** | 16.36M | 3.1197 | **1.2077** | 11,073 | 54ms | 15.35 MB | **Yes** |

**PyTorch 2.6, SP8192, 10-min wall clock. PG baseline: 1.2244 BPB.**

**Best submittable run: Run 11 (SP8192 combo slim + TTT)** — 1.2077 BPB, **-0.0167 below PG baseline**, 15.35 MB (under budget with 0.52 MB headroom). First run that beats baseline AND fits in 16 MB.

### 3-Seed Reproducibility — SP8192 Combo Slim + TTT, 8×H100

| Seed | val_loss (TTT) | val_bpb (TTT) | Steps | Size (int8+zlib) |
|------|----------------|---------------|-------|-------------------|
| 42 | 3.1200 | 1.2078 | 11,028 | 15.34 MB |
| 1337 | 3.1169 | **1.2067** | 11,030 | 15.34 MB |
| 2025 | 3.1190 | 1.2075 | 11,036 | 15.34 MB |
| **Mean** | **3.1186** | **1.2073** | **11,031** | — |
| **Std** | — | **±0.0006** | — | — |

**Confirms robustness:** 3 seeds produce nearly identical results (std 0.0006 BPB). Mean 1.2073 ≈ Run 11's 1.2077 — reproducible within noise.

## Experiment Runs — 2×H100 (sorted by BPB, best first)

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Size (int8+zlib) | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------------------|---------|
| 13†‡ | SP8192 combo slim + TTT (re-run) | 16.36M | 3.1990 | 1.2384 | 2,749 | 218ms | 15.09 MB | Yes |
| D†‡ | SP8192 combo slim + TTT (re-run) | 16.36M | 3.2021 | 1.2396 | 2,652 | 226ms | 15.08 MB | Yes |
| A† | SP8192 combo slim + TTT | 16.36M | 3.2059 | 1.2411 | 2,572 | 233ms | 15.03 MB | Yes |
| H† | SP8192 combo slim (no TTT) | 16.36M | 3.2112 | 1.2432 | 2,541 | 236ms | 15.04 MB | Yes |
| 3 | Elementwise gated attn* | 19.42M | 2.1280 | 1.2602 | 3,129 | 192ms | 17.87 MB | **No** |
| 7 | LeakyReLU² | 17.06M | 2.1344 | 1.2641 | 3,673 | 163ms | 15.77 MB | Yes |
| 8 | LeakyReLU² + headwise* | 17.10M | 2.1345 | 1.2642 | 3,368 | 178ms | 15.77 MB | Yes |
| 6v2 | Baseline repeat | 17.06M | 2.1357 | 1.2649 | 3,661 | 164ms | 15.77 MB | Yes |
| 2 | Headwise gated attn* | 17.10M | 2.1366 | 1.2653 | 3,287 | 182ms | 15.75 MB | Yes |
| 6 | Baseline (GQA) | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | 15.75 MB | Yes |
| 12 | Baseline (PyTorch 2.6) | 17.06M | 2.1462 | 1.2711 | 3,087 | 194ms | 15.70 MB | Yes |
| 9 | Headwise + QK-Gain 5.0 | 17.10M | 2.1475 | 1.2719 | 2,861 | 210ms | 15.65 MB | Yes |
| 4 | MQA (1 KV head) | 17.65M | 2.1549 | 1.2761 | 3,370 | 178ms | 16.84 MB | **No** |
| ~~5~~ | ~~INVALID (stale env)~~ | — | — | — | — | — | — | — |

**Runs 2-9, 12: SP1024, 10-min wall clock. Runs 2-9: PyTorch 2.11. Run 12: PyTorch 2.6 (18% slower per step). † Runs A, D, H, 13: SP8192, 2×H100, 2026-04-26.**

**‡ SLM INVALID:** Runs D and 13 originally claimed SLM, but the RunPod had commit `d7af1ec` (Apr 23) which predates the SLM code push (Apr 26 7:18 PM). `SLM_ENABLED` env var was set but ignored — the code to use it didn't exist yet. BPB values are valid as additional Run A repeats (SP8192 combo slim + TTT). Run-to-run variance: A=1.2411, D=1.2396, 13=1.2384 (spread 0.0027, consistent with noise). **SLM validation pending — Session 7.**

> **Note:** Run 1 (2026-04-16) excluded — ran on 1×GPU (old pod, PyTorch 2.4.1), completed only 1,819 steps. Result (1.3045 BPB) is not comparable. Run 12 was an accidental vanilla baseline (config source failed, intended as SP8192 combo slim).

*\* Original technique by James Vo — gated attention applied post-SDPA with sigmoid gates, inspired by NeurIPS 2025 Best Paper (arxiv.org/abs/2505.06708).*

### Session 6 — SLM Ratio Sweep & Technique Stacking (SP1024, 1-GPU†)

> **⚠️ NGPUS bug:** These SP1024 runs used 1 GPU due to an env var overwrite bug (step_avg ~220ms vs normal ~163ms on 2×H100). Absolute BPB is inflated but **relative comparisons are valid.**
>
> **⚠️ SLM INVALID:** The RunPod had commit `d7af1ec` (Apr 23) which predates the SLM code push (Apr 26 7:18 PM). ALL "SLM" runs below were actually running without SLM — the `SLM_ENABLED` env var was set but the code to use it didn't exist yet. BPB differences in the "SLM sweep" are run-to-run noise. Technique stacking runs labeled "+SLM" were actually running their base technique only. **SLM validation pending — Session 7.**

**SLM Ratio Sweep** (SP1024, GQA baseline + SLM):

| SLM k | val_bpb | Steps | Notes |
|-------|---------|-------|-------|
| 0.4 | 1.2954 | 2,784 | Too aggressive — skips 60% of tokens |
| 0.5 | 1.3001 | 2,554 | Worst — too many tokens dropped |
| 0.6 | 1.2994 | 2,615 | Paper-recommended range |
| 0.7 | 1.2973 | 2,713 | |
| **0.8** | **1.2949** | **2,801** | **Best ratio — more conservative than paper** |
| 0.9 | 1.2955 | 2,818 | Nearly full training |

**Technique Stacking** (SP1024, 1-GPU):

| Technique | val_bpb | Steps | Notes |
|-----------|---------|-------|-------|
| LeakyReLU² + headwise + SLM k=0.6 | **1.2899** | 2,933 | Best SP1024 combo |
| LeakyReLU² + SLM k=0.6 | 1.2927 | 2,655 | |
| Headwise + QKG5 + SLM k=0.6 | 1.2927 | 2,939 | |
| LeakyReLU² | 1.2932 | 2,619 | |
| Headwise + QK-Gain 5.0 | 1.2981 | 2,668 | |
| SLM test k=0.6 | 1.2994 | 2,615 | |

**Key findings:** (1) ~~SLM k=0.8 optimal at 17M scale~~ **INVALID — SLM code not present, see note above.** (2) Techniques stack: LReLU²+headwise beats any single technique (SLM contribution was noise, not real). (3) SP8192 still dominates: best SP1024 combo (1.2899 on 1-GPU) can't match SP8192 on same hardware (1.2411-1.2432).

† All runs 2026-04-26, PyTorch 2.6, 10-min wall clock. NGPUS bug confirmed by step_avg (~220ms vs expected ~163ms for 2×H100).

---

## Competition Context

**Challenge:** Train the best language model in ≤ 10 min on 8×H100 SXM. Artifact ≤ 16 MB (16,000,000 bytes, decimal — code + compressed model). Scored by FineWeb validation BPB (bits per byte, tokenizer-agnostic). New SOTA must beat previous by ≥ 0.005 nats at p < 0.01.

**Naive Baseline:** 1.2244 BPB — 9L, 512 dim, 1024 vocab, tied embeddings, 4 KV heads, int8+zlib.

**Current SOTA:** 1.0810 BPB — SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal score-first TTT (bigbag, 2026-04-09).

### Official Leaderboard (as of 2026-04-09)

| Rank | BPB | Author | Key Techniques |
|-----:|------:|--------|---------------|
| 1 | 1.0810 | bigbag | SP8192, 3-layer recurrence, parallel residuals, QK-Gain 5.25, legal TTT |
| 2 | 1.0822 | aryanbhosale | SP8192, parallel residuals, score-first TTT |
| 3 | 1.0828 | dexhunter | SP8192, QK-Gain 5.0, legal score-first TTT |
| 4 | 1.0835 | Robby Sneiderman | SP8192, parallel residuals, Hessian-aware SDClip, progressive recurrence |
| 5 | 1.0856 | Kevin Clark | SP8192, GPTQ embeddings, looped layers 4-5, MuonEq-R, SDClip |
| 6 | 1.0897 | aryanbhosale | SP4096, depth recurrence, parallel residuals, MuonEq-R, QK-Gain 5.0 |
| 7 | 1.0912 | dexhunter | MuonEq-R, depth recurrence, WD=0.090, all-int6 GPTQ |
| 8 | 1.0979 | Kevin Clark | SP4096, 4x MLP, high WD (removed TTT/hash embed/SmearGate) |
| 9 | 1.1063 | Marko Sisovic | Parallel residuals, mini depth recurrence (layers 4-5), AR self-gen GPTQ |
| 10 | 1.1147 | abaybektursun | Self-gen GPTQ calibration, all-layer XSA |
| 11 | 1.1194 | abaybektursun | LeakyReLU², legal score-first TTT, parallel Muon |
| 12 | 1.1228 | signalrush | 11L, EMA, GPTQ-lite, warmdown3500, QAT@0.15 |
| 13 | 1.1248 | jfprincz | 11L, partial RoPE (16/64), LN scale, EMA, XSA4 |
| 14 | 1.1271 | jfprincz | 11L, XSA4, EMA, int6, MLP3x |
| 15 | 1.1307 | unnir | 11L, efficient partial XSA, FA3, SWA120 |
| 16 | 1.1428 | thwu1 | 10L, mixed int5/int6, BigramHash(10240), SWA |
| 17 | 1.1458 | Raahil Shah | Int6, MLP3x, SmearGate, BigramHash, OrthoInit, Muon WD, SWA |
| 18 | 1.1502 | aruniyer | 11L, MLP3x, int6 QAT, zstd-22, sliding eval |
| 19 | 1.1556 | aquariouseworkman | SmearGate, BigramHash, 3x MLP, int6 STE QAT, sliding eval |
| 20 | 1.1570 | Ciprian-Florin Ifrim | 73.7M params, ternary quantization (1/0/-1) |
| 21 | 1.1586 | yahya010 | 10L, int6 QAT + zstd-22, MLP 1344, Muon 0.99, sliding eval |
| 22 | 1.1630 | aquariouseworkman | Int6 blocks + int8 embeds, 3x MLP, sliding eval |
| 23 | 1.1748 | notapplica | Spectral embed init, resid mix, 10L, Muon WD |
| 24 | 1.1925 | Matthew Li | Sliding window eval (stride=64) |
| 25 | 1.1928 | samacqua | LoRA TTT |
| 26 | 1.2014 | Spokane Way | 4k seq length + tuned hypers |
| 27 | 1.2060 | Spokane Way | 2048 seq length (train + val) |
| **→** | **1.2077** | **Us (Run 11)** | **SP8192 combo slim + TTT, MODEL_DIM=448** |
| 28 | 1.2147 | Nan Liu | 10L, mixed int8/int6 |
| 29 | 1.2197 | Renier Velazco | FP16 tied embedding, LR/warmdown tuning |
| 30 | 1.2244 | Baseline | 9L, 512 dim, 1024 vocab, tied embeddings, 4 KV heads |

### Where We Stand

- **Our best (Run 11):** 1.2077 BPB — would rank **~28th** of 30 entries
- **Gap to baseline:** −0.0167 BPB (beats it, submittable)
- **Gap to SOTA (#1):** +0.1267 BPB
- **Gap to nearest technique match (#11, LeakyReLU² + TTT):** +0.0883 BPB — that entry also uses parallel Muon, self-gen GPTQ, and XSA which we haven't tried
- **Biggest leaderboard patterns:** Top 5 all use SP8192 + depth recurrence; top 8 all use advanced quantization (GPTQ/int6); ranks 10-15 all use XSA and/or EMA

---

## Run Log

### Run 1 — 2-GPU baseline (2026-04-16)

| Item | Value |
|---|---|
| Date | 2026-04-16 |
| GPUs | 2× (auto-detected, intended 1) |
| Pod cost | **~$2** (< 1 hour) |
| Model params | 17,059,912 (~17M) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 1,819 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,286 MiB |
| **val_loss (int8+zlib)** | **2.2027** (computed) |
| **val_bpb (int8+zlib)** | **1.3045** |
| Baseline to beat | 1.2244 |
| Gap | +0.0801 (need to go lower) |
| Model size (raw) | 67.2 MB |
| Model size (int8+zlib) | **14.7 MB** (under 16 MB budget) |
| Compression ratio | 3.91× |

**Training curve:**
- Step 0: val_bpb 4.1077 (near random)
- Step 1000: val_bpb 1.3702
- Step 1819: val_bpb 1.3033 (still improving when time ran out)

**Observations:**
- Model was still improving at cutoff — only used 2 GPUs, so throughput was ~4× lower than the intended 8-GPU setup
- int8+zlib compression fits comfortably under 16 MB
- BPB roundtrip after compression: 1.3045 (negligible degradation from 1.3033)

### Run 2 — Gated Attention headwise, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | Gated Attention — headwise (1 sigmoid gate per head, NeurIPS 2025 Best Paper) |
| Paper | arxiv.org/abs/2505.06708 |
| Model params | 17,096,776 (~17.1M, +37K from gate projections) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,287 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,374 MiB |
| Step avg | 182.54 ms |
| **val_loss (raw)** | **2.1320** (computed) |
| **val_bpb (raw)** | **1.2626** |
| **val_loss (int8+zlib)** | **2.1366** (computed) |
| **val_bpb (int8+zlib)** | **1.2653** |
| Baseline to beat | 1.2244 |
| Gap | +0.0409 (down from +0.0801 in Run 1) |
| Model size (raw) | 67.4 MB |
| Model size (int8+zlib) | **15.75 MB** (under 16 MB budget) |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1085
- Step 1000: val_bpb 1.3780
- Step 2000: val_bpb 1.3208
- Step 3000: val_bpb 1.2725
- Step 3287: val_bpb 1.2626 (still improving when time ran out)

**Observations:**
- Gap to baseline cut in half vs Run 1 (0.0409 vs 0.0801)
- Much higher throughput than Run 1 (3,287 vs 1,819 steps) — likely due to PyTorch 2.11.0 vs 2.4.1 on previous pod
- Headwise gates add only ~37K params (~0.2% overhead) — nearly free
- Model still improving at cutoff — not converged
- Compressed size 15.75 MB, tight but under 16 MB budget

**Note:** Run 1 vs Run 2 improvement comes from two factors: (1) gated attention headwise, and (2) newer PyTorch (2.11 vs 2.4.1) giving higher throughput. Need a clean baseline on the same pod/PyTorch to isolate the gated attention effect.

### Run 3 — Gated Attention elementwise, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | Gated Attention — elementwise (1 sigmoid gate per dim per head, NeurIPS 2025 Best Paper) |
| Paper | arxiv.org/abs/2505.06708 |
| Model params | 19,419,208 (~19.4M, +2.36M from gate projections) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,129 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 11,518 MiB |
| Step avg | 191.75 ms |
| **val_loss (raw)** | **2.1241** (computed) |
| **val_bpb (raw)** | **1.2579** |
| **val_loss (int8+zlib)** | **2.1280** (computed) |
| **val_bpb (int8+zlib)** | **1.2602** |
| Baseline to beat | 1.2244 |
| Gap | +0.0358 |
| Model size (raw) | 76.7 MB |
| Model size (int8+zlib) | **17.87 MB (OVER 16 MB budget)** |
| Compression ratio | 3.92× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1098
- Step 1000: val_bpb 1.3696
- Step 2000: val_bpb 1.3130
- Step 3000: val_bpb 1.2613
- Step 3129: val_bpb 1.2579 (still improving when time ran out)

**Observations:**
- Slightly better BPB than headwise (1.2602 vs 1.2653) but **busts the 16 MB budget** (17.87 MB)
- 2.36M extra params is too many — compressed model 1.87 MB over limit
- Slower per step (191ms vs 182ms headwise) → fewer steps (3,129 vs 3,287)
- More VRAM (11.5 GB vs 10.4 GB)
- The marginal BPB gain (~0.005) is not worth the budget/speed cost

### Run 2 vs Run 3 — Headwise vs Elementwise (same pod, same PyTorch)

| Metric | Headwise (Run 2) | Elementwise (Run 3) | Winner |
|---|---|---|---|
| val_bpb (compressed) | 1.2653 | 1.2602 | Elementwise (+0.005) |
| Extra params | +37K (0.2%) | +2.36M (14%) | Headwise |
| Compressed size | 15.75 MB | 17.87 MB | Headwise (under budget) |
| Steps completed | 3,287 | 3,129 | Headwise |
| Step avg | 182 ms | 192 ms | Headwise |
| Peak VRAM | 10.4 GB | 11.5 GB | Headwise |
| Under 16 MB? | Yes | **No** | Headwise |

**Verdict:** Headwise wins for PG. Elementwise has marginally better BPB but fails the size constraint and is slower. Headwise is nearly free (37K params, no speed penalty) and fits under budget.

### Run 4 — MQA (NUM_KV_HEADS=1), 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | MQA — Multi-Query Attention (1 KV head shared across all 8 Q heads) |
| Model params | 17,649,736 (~17.6M) |
| Architecture | 9 layers, 512 dims, 8 heads, **1 kv_head** (MQA) |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,370 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,710 MiB |
| Step avg | 178.03 ms |
| **val_loss (raw)** | **2.1503** (computed) |
| **val_bpb (raw)** | **1.2734** |
| **val_loss (int8+zlib)** | **2.1549** (computed) |
| **val_bpb (int8+zlib)** | **1.2761** |
| Baseline to beat | 1.2244 |
| Gap | +0.0517 |
| Model size (raw) | 69.6 MB |
| Model size (int8+zlib) | **16.84 MB (OVER 16 MB budget)** |
| Compression ratio | 3.79× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1085
- Step 1000: val_bpb 1.3864
- Step 2000: val_bpb 1.3329
- Step 3000: val_bpb 1.2861
- Step 3370: val_bpb 1.2734 (still improving when time ran out)

**Observations:**
- Worst BPB of the three techniques tested on this pod (1.2761 vs headwise 1.2653 vs elementwise 1.2602)
- Also over 16 MB budget (16.84 MB)
- Fastest per step (178ms) — fewer KV params = less compute, more steps (3,370)
- Despite more steps, BPB is worse — MQA trades quality for speed, bad tradeoff for PG where BPB matters

### All Runs on PyTorch 2.11 Pod — Comparison

| Run | Technique | KV Heads | val_loss | val_bpb | Steps | Size (int8+zlib) | Under 16 MB? |
|-----|-----------|----------|----------|---------|-------|-------------------|-------------|
| 2 | Gated Attn (headwise) | 4 (GQA) | 2.1366 | **1.2653** | 3,287 | **15.75 MB** | **Yes** |
| 3 | Gated Attn (elementwise) | 4 (GQA) | 2.1280 | 1.2602 | 3,129 | 17.87 MB | No |
| 4 | MQA | 1 (MQA) | 2.1549 | 1.2761 | 3,370 | 16.84 MB | No |

### Run 5 — GQA baseline (intended), 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline (no gated attention) — **INVALID: stale env var** |
| Model params | 19,419,208 (~19.4M) — should be 17M, shell had `GATED_ATTN=elementwise` from prior run |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,296 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 11,518 MiB |
| Step avg | 182.03 ms |
| **val_loss (raw)** | **2.1191** |
| **val_bpb (raw)** | **1.2550** |
| **val_loss (int8+zlib)** | **2.1234** |
| **val_bpb (int8+zlib)** | **1.2576** |
| Baseline to beat | 1.2244 |
| Gap | +0.0332 |
| Model size (raw) | 76.66 MB |
| Model size (int8+zlib) | **17.96 MB (OVER 16 MB budget)** |
| Compression ratio | 3.92× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1098
- Step 1000: val_bpb 1.3687
- Step 2000: val_bpb 1.3144
- Step 3000: val_bpb 1.2648
- Step 3296: val_bpb 1.2550 (still improving when time ran out)

**Observations:**
- **INVALID RUN:** Intended as a clean GQA baseline, but shell had stale `GATED_ATTN=elementwise` env var from a previous `source runs/configs/gated_attn_elementwise.env`. Param count (19.4M) matches Run 3 exactly.
- Results are essentially Run 3 re-run with slightly more steps (3,296 vs 3,129) → slightly better BPB (1.2576 vs 1.2602)
- Confirms elementwise model over 16 MB budget (17.96 MB)
- **Root cause fix:** All env configs now explicitly set `GATED_ATTN=none` and `ACTIVATION=relu2` as defaults, so sourcing any config resets stale env vars. Clean baseline should have 17,059,912 params and ~14.7 MB compressed.

### Run 6 — Clean GQA baseline, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline (no gated attention, no activation change) — clean re-run after env var fix |
| Model params | 17,059,912 (~17M) — confirmed correct |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,500 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | 171.41 ms |
| **val_loss (raw)** | **2.1338** |
| **val_bpb (raw)** | **1.2638** |
| **val_loss (int8+zlib)** | **2.1388** |
| **val_bpb (int8+zlib)** | **1.2667** |
| Baseline to beat | 1.2244 |
| Gap | +0.0423 |
| Model size (raw) | 67.22 MB |
| Model size (int8+zlib) | **15.75 MB (under 16 MB budget, +0.14 MB headroom)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 0: val_bpb 4.1077
- Step 1000: val_bpb 1.3830
- Step 2000: val_bpb 1.3239
- Step 3000: val_bpb 1.2812
- Step 3500: val_bpb 1.2638 (still improving when time ran out)

**Observations:**
- Env var fix confirmed: 17,059,912 params (correct), not 19.4M
- Budget check script now shows int8+zlib size — 15.75 MB, under budget with 0.14 MB headroom
- More steps than Run 2 (3,500 vs 3,287) — no gate overhead means faster per step (171ms vs 182ms)
- But worse BPB than Run 2 headwise (1.2667 vs 1.2653) despite more steps — confirms headwise gated attention genuinely helps quality
- This is the proper control for isolating technique effects on this pod

### All Runs on PyTorch 2.11 Pod — Updated Comparison

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Size (int8+zlib) | Under 16 MB? |
|-----|-----------|--------|----------|---------|-------|----------|-------------------|-------------|
| 6 | **Baseline (GQA)** | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | 15.75 MB | Yes |
| 2 | Gated Attn (headwise) | 17.10M | 2.1366 | **1.2653** | 3,287 | 182ms | 15.75 MB | **Yes** |
| 3 | Gated Attn (elementwise) | 19.42M | 2.1280 | 1.2602 | 3,129 | 192ms | 17.87 MB | No |
| 4 | MQA | 17.65M | 2.1549 | 1.2761 | 3,370 | 178ms | 16.84 MB | No |

**Key finding:** Headwise gated attention improves BPB (1.2653 vs 1.2667 baseline) despite fewer steps (3,287 vs 3,500). The quality gain outweighs the speed cost. Elementwise has even better BPB but busts the budget.

### Run 6v2 — Clean GQA baseline repeat, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | GQA baseline repeat — reproducibility check after env var fix + branch sync |
| Model params | 17,059,912 (~17M) — confirmed correct |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,661 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | 163.87 ms |
| **val_loss (raw)** | **2.1304** |
| **val_bpb (raw)** | **1.2617** |
| **val_loss (int8+zlib)** | **2.1357** |
| **val_bpb (int8+zlib)** | **1.2649** |
| Baseline to beat | 1.2244 |
| Gap | +0.0405 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget, +0.12 MB headroom)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- Reproducibility confirmed: very close to Run 6 (1.2649 vs 1.2667, within noise)
- Faster step time than Run 6 (164ms vs 171ms) — GPU thermal variance between runs
- More steps (3,661 vs 3,500) → slightly better BPB
- Used as the control baseline for Runs 7-8 ablation

### Run 7 — LeakyReLU², 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | LeakyReLU(0.5)² activation (PG ranks 10-11) |
| Model params | 17,059,912 (~17M) — no extra params (LeakyReLU has no learnable weights) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,673 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,777 MiB |
| Step avg | ~163 ms |
| **val_loss (int8+zlib)** | **2.1344** |
| **val_bpb (int8+zlib)** | **1.2641** |
| Baseline to beat | 1.2244 |
| Gap | +0.0397 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- LeakyReLU² gives **slightly better BPB than baseline** (1.2641 vs 1.2649 baseline_v2) — confirms leaderboard finding that it improves per-step quality
- Same param count as baseline — activation change is free
- Same step speed as baseline (~163 ms) — no throughput penalty
- Under budget (15.77 MB)
- Improvement is small (+0.0008 BPB) but consistent with ranks 10-11 using this technique

### Run 8 — LeakyReLU² + Headwise Gated Attn, 2×H100 (2026-04-23)

| Item | Value |
|---|---|
| Date | 2026-04-23 |
| GPUs | 2× H100 |
| Technique | LeakyReLU(0.5)² + Headwise Gated Attention (combo test) |
| Model params | 17,096,776 (~17.1M, +37K from headwise gates) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 3,368 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 10,824 MiB |
| Step avg | 178.15 ms |
| **val_loss (raw)** | **2.1294** |
| **val_bpb (raw)** | **1.2611** |
| **val_loss (int8+zlib)** | **2.1345** |
| **val_bpb (int8+zlib)** | **1.2642** |
| Baseline to beat | 1.2244 |
| Gap | +0.0398 |
| Model size (int8+zlib) | **15.77 MB (under 16 MB budget)** |
| Compression ratio | 3.91× |
| PyTorch version | 2.11.0 |

**Observations:**
- Combo of LeakyReLU² + headwise gives **1.2642** — essentially identical to LeakyReLU² alone (1.2641)
- Headwise adds 178ms/step overhead vs 163ms baseline → fewer steps (3,368 vs 3,673)
- The techniques don't stack — LeakyReLU² is doing the work, headwise doesn't add value on top
- Under budget (15.77 MB)

### Runs 6-8 — Activation & Gated Attn Ablation (same pod, same PyTorch)

| Run | Technique | val_loss | val_bpb | Steps | Step avg | Size |
|-----|-----------|----------|---------|-------|----------|------|
| 6v2 | Baseline | 2.1357 | 1.2649 | 3,661 | 164ms | 15.77 MB |
| 7 | LeakyReLU² | 2.1344 | **1.2641** | 3,673 | 163ms | 15.77 MB |
| 8 | LeakyReLU² + headwise | 2.1345 | 1.2642 | 3,368 | 178ms | 15.77 MB |

**Key finding:** LeakyReLU² gives a small free improvement (-0.0008 BPB, no speed/size cost). Adding headwise gates on top doesn't help — the speed penalty (fewer steps) negates any per-step quality gain.

### Run 9 — Headwise + QK-Gain 5.0, 2×H100 (2026-04-24)

| Item | Value |
|---|---|
| Date | 2026-04-24 |
| GPUs | 2× H100 |
| Technique | Headwise gated attention + QK-Gain 5.0 (PG ranks 1-6 use 5.0-5.25) |
| Model params | 17,096,776 (~17.1M, +37K from headwise gates) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 1024 (SentencePiece BPE) |
| Batch tokens | 524,288 |
| Grad accum steps | 4 |
| Warmup steps | 20 |
| Steps completed | 2,861 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 13,217 MiB |
| Step avg | 210.48 ms |
| **val_loss (raw)** | **2.1438** |
| **val_bpb (raw)** | **1.2697** |
| **val_loss (int8+zlib)** | **2.1475** |
| **val_bpb (int8+zlib)** | **1.2719** |
| Baseline to beat | 1.2244 |
| Gap | +0.0475 |
| Model size (int8+zlib) | **15.65 MB (under 16 MB budget, +0.24 MB headroom)** |
| PyTorch version | 2.11.0 |

**Training curve:**
- Step 2700: train_loss 2.1388
- Step 2800: val_bpb 1.2709
- Step 2861: val_bpb 1.2697 (stopped at wall clock cap)

**Observations:**
- **QK-Gain 5.0 hurt on SP1024** — val_bpb 1.2719 vs 1.2653 (headwise alone, Run 2)
- Much slower per step: 210ms vs 182ms (headwise alone) — 15% slower
- Fewer steps completed: 2,861 vs 3,287 (headwise alone) — 13% fewer
- Higher VRAM: 13,217 MiB vs 10,374 MiB (headwise alone)
- QK-Gain 5.0 likely needs SP8192 to be effective — all leaderboard uses of QK-Gain 5.0+ are SP8192
- The speed penalty + VRAM overhead negate any potential per-step quality gain at SP1024 scale

### Run 10 — SP8192 Combo + Score-First TTT, 8×H100 (2026-04-24)

| Item | Value |
|---|---|
| Date | 2026-04-24 |
| GPUs | 8× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise gated attn |
| Model params | 20,766,792 (~20.8M) |
| Architecture | 9 layers, 512 dims, 8 heads, 4 kv_heads, GQA |
| Vocab | 8192 (SentencePiece BPE, from kevclark/parameter-golf) |
| Batch tokens | 524,288 |
| Steps completed | 10,582 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 16,507 MiB |
| Step avg | 56.71 ms |
| **val_loss (raw)** | **3.0517** |
| **val_bpb (raw)** | **1.1814** |
| **val_loss (int8+zlib)** | **3.0718** |
| **val_bpb (int8+zlib)** | **1.1892** |
| **val_loss (TTT)** | **3.0666** |
| **val_bpb (TTT)** | **1.1872** |
| TTT eval time | 163.6s (1,238 chunks × 32K tokens, SGD lr=0.005, 3 epochs/chunk) |
| PG baseline | 1.2244 |
| Gap | **-0.0372** (BELOW baseline!) |
| Model size (int8+zlib) | **19.41 MB (OVER 16 MB budget by 3.53 MB)** |
| Compression ratio | 3.57× |
| PyTorch version | 2.6.0 |

**Training curve:**
- Step 0: training starts
- Step 10,500: train_loss 3.2411
- Step 10,582: val_bpb 1.1814 (stopped at wall clock cap)

**TTT progression (running BPB over 1,238 chunks):**
- Chunk 1: 1.1963
- Chunk 111: 1.1864 (best early)
- Chunk 881: 1.1843 (best overall)
- Chunk 1238: **1.1872** (final — slight regression from peak due to later validation chunks)

**Observations:**
- **FIRST RUN TO BEAT PG BASELINE** — 1.1872 vs 1.2244, gap -0.0372
- SP8192 is the dominant factor: 1.1892 (int8) vs 1.2641 (best SP1024) = -0.075 BPB improvement
- 8×H100 gives 3× more steps than 2×H100 (10,582 vs ~3,500) in same 10-min wall clock
- Score-First TTT improves BPB by -0.002 over non-TTT (1.1892 → 1.1872)
- **Budget fail: 19.41 MB > 16 MB** — SP8192 embedding table adds ~3.5M params (8192×512 vs 1024×512)
- val_loss is ~3.05 (vs ~2.13 for SP1024) — expected since SP8192 has more possible tokens per position
- QK-Gain 5.0 works well with SP8192 (unlike SP1024 where it hurt, Run 9)
- Need to reduce MODEL_DIM from 512 to ~448 to fit budget, or use GPTQ embedding quantization

### Run 11 — SP8192 Combo Slim + Score-First TTT, 8×H100 (2026-04-24)

| Item | Value |
|---|---|
| Date | 2026-04-24 |
| GPUs | 8× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + **MODEL_DIM=448** |
| Model params | 16,364,616 (~16.4M) |
| Architecture | 9 layers, **448 dims**, 8 heads, 4 kv_heads, GQA |
| Vocab | 8192 (SentencePiece BPE, from kevclark/parameter-golf) |
| Steps completed | 11,073 / 20,000 (hit 10-min wall clock cap) |
| Peak VRAM | 15,287 MiB |
| Step avg | 54.19 ms |
| **val_loss (raw)** | **3.1053** |
| **val_bpb (raw)** | **1.2022** |
| **val_loss (int8+zlib)** | **3.1245** |
| **val_bpb (int8+zlib)** | **1.2096** |
| **val_loss (TTT)** | **3.1197** |
| **val_bpb (TTT)** | **1.2077** |
| TTT eval time | 160.8s |
| PG baseline | 1.2244 |
| Gap | **-0.0167** (BELOW baseline!) |
| Model size (int8+zlib) | **15.35 MB (under 16 MB budget, +0.52 MB headroom)** |
| Compression ratio | 3.53× |
| PyTorch version | 2.6.0 |

**Observations:**
- **FIRST SUBMITTABLE RUN** — beats PG baseline (1.2077 vs 1.2244) AND fits in 16 MB
- MODEL_DIM reduction (512→448) cost +0.020 BPB (1.2077 vs 1.1872) but saved 4.06 MB
- More steps than Run 10 (11,073 vs 10,582) — smaller model trains faster per step (54ms vs 57ms)
- TTT improved BPB by -0.002 over non-TTT (1.2096 → 1.2077)
- 0.52 MB headroom — room for a slightly wider model (maybe MODEL_DIM=464?)

### Run A — SP8192 Combo Slim + TTT (retry), 2×H100 (2026-04-26)

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 |
| Params | 16,364,616 (~16.4M) |
| Vocab | 8192 (SentencePiece BPE) |
| Steps completed | 2,572 / 20,000 (hit 10-min wall clock cap) |
| Step avg | 233 ms |
| Peak VRAM | 10,570 MiB |
| **val_bpb (int8+zlib)** | **1.2420** |
| **val_bpb (TTT)** | **1.2411** |
| TTT eval time | 144s (1,238 chunks) |
| Model size (int8+zlib) | **15.03 MB** (under 16 MB budget) |
| PyTorch version | 2.6.0 |

**Observations:**
- Confirms Run 11 config works on 2×H100 — 1.2411 vs Run 11's 1.2077 (fewer steps → higher BPB, as expected)
- TTT contribution: -0.0009 BPB (1.2420 → 1.2411)

### Run D — SP8192 Combo Slim + TTT (re-run), 2×H100 (2026-04-26)

> **⚠️ SLM INVALID:** Originally logged as "SLM k=0.6" but RunPod had commit `d7af1ec` (no SLM code). This run is a valid Run A repeat.

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 (SLM was set but code absent) |
| Params | 16,364,616 (~16.4M) |
| Vocab | 8192 (SentencePiece BPE) |
| Steps completed | 2,652 / 20,000 (hit 10-min wall clock cap) |
| Step avg | 226 ms |
| Peak VRAM | 10,570 MiB |
| **val_bpb (int8+zlib)** | **1.2408** |
| **val_bpb (TTT)** | **1.2396** |
| TTT eval time | 145s (1,238 chunks) |
| Model size (int8+zlib) | **15.08 MB** (under 16 MB budget) |
| PyTorch version | 2.6.0 |

**Observations:**
- ~~SLM k=0.6 improves over no-SLM: 1.2396 vs Run A's 1.2411 (-0.0015)~~ **INVALID** — SLM code not present; difference is run-to-run noise
- ~~More steps than Run A (2,652 vs 2,572) — SLM skips easy tokens~~ Step count variance is normal (±100 steps between identical runs)

### Run H — SP8192 Combo Slim (no TTT ablation), 2×H100 (2026-04-26)

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 (NO TTT, NO SLM) |
| Params | 16,364,616 (~16.4M) |
| Vocab | 8192 (SentencePiece BPE) |
| Steps completed | 2,541 / 20,000 (hit 10-min wall clock cap) |
| Step avg | 236 ms |
| Peak VRAM | 10,570 MiB |
| **val_bpb (int8+zlib)** | **1.2432** |
| Model size (int8+zlib) | **15.04 MB** (under 16 MB budget) |
| PyTorch version | 2.6.0 |

**Observations:**
- TTT ablation: Run H (no TTT) 1.2432 vs Run A (TTT) 1.2411 — **TTT contributes -0.0021 BPB**
- Confirms TTT is worth the eval-time cost even on 2×H100

### Run 13 — SP8192 Combo Slim + TTT (re-run), 2×H100 (2026-04-26)

> **⚠️ SLM INVALID:** Originally logged as "SLM k=0.8" but RunPod had commit `d7af1ec` (no SLM code). This run is a valid Run A repeat.

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 (SLM was set but code absent) |
| Params | 16,364,616 (~16.4M) |
| Vocab | 8192 (SentencePiece BPE) |
| Steps completed | 2,749 / 20,000 (hit 10-min wall clock cap) |
| Step avg | 218 ms |
| **val_bpb (TTT)** | **1.2384** |
| Model size (int8+zlib) | **15.09 MB** (under 16 MB budget) |
| PyTorch version | 2.6.0 |

**Observations:**
- ~~**Best 2×H100 run** — SLM k=0.8 is optimal ratio from sweep~~ **INVALID** — SLM code not present
- BPB variance across Run A repeats: A=1.2411, D=1.2396, 13=1.2384 (spread ±0.0014, normal noise)
- Step count variance: A=2,572, D=2,652, 13=2,749 (spread ±89 steps, normal)

### Runs A–H–D–13 — SP8192 2×H100 Ablation Summary

> **⚠️ SLM INVALID for D and 13** — see ‡ note in summary table. SLM deltas below are noise, not real.

| Run | TTT | SLM (claimed) | val_bpb | TTT Δ | SLM Δ |
|-----|-----|---------------|---------|-------|-------|
| H | No | No | 1.2432 | — | — |
| A | Yes | No | 1.2411 | -0.0021 | — |
| D‡ | Yes | ~~k=0.6~~ none | 1.2396 | -0.0021 | ~~-0.0015~~ noise |
| 13‡ | Yes | ~~k=0.8~~ none | 1.2384 | -0.0021 | ~~-0.0027~~ noise |

## Techniques That Worked

_Add entries as we discover things._

| Technique | Impact on BPB | Why it works |
|---|---|---|
| **SP8192 vocab scaling** | 1.2411 (2×H100) vs 1.2649 SP1024 = **-0.024 BPB** | Bigger vocab = fewer tokens per byte = lower BPB. Single biggest lever. All top 8 leaderboard entries use SP4096/SP8192. Paper: "Scaling Laws with Vocabulary" (NeurIPS 2024). |
| ~~SLM k=0.8 (Rho-1)~~ | ~~-0.0027 BPB on SP8192~~ **INVALID** | SLM code was not present on RunPod during Session 6. Validation pending — Session 7. Paper: "Not All Tokens Are What You Need" (NeurIPS 2024). |
| **Score-First TTT** | -0.0021 BPB (1.2432→1.2411) | Eval-time fine-tuning: adapt model to validation distribution. Legal under PG rules (score before update). Source: @dexhunter (PG competition). |
| Gated Attention (headwise) | 1.2653 (compressed), fits 16 MB | Sigmoid gate after SDPA lets model suppress uninformative heads per token. Nearly free: +37K params, no speed penalty. Original technique by James Vo. |
| LeakyReLU² | -0.0008 BPB vs baseline (1.2641 vs 1.2649) | Free activation swap — no extra params, no speed cost. Used by PG ranks 10-11. |

## Techniques That Didn't Work

| Technique | Expected impact | Actual result | Why it failed |
|---|---|---|---|
| Gated Attention (elementwise) | Better BPB than headwise | 1.2602 BPB but 17.87 MB (over budget) | +2.36M params makes compressed model too large. Marginal BPB gain (+0.005) not worth the cost. |
| QK-Gain 5.0 (on SP1024) | Better attention scaling | 1.2719 BPB (worse than headwise 1.2653) | 15% slower steps (210ms vs 182ms), higher VRAM (13GB vs 10GB). QK-Gain 5.0 likely needs SP8192 to be effective. |

## Key Insights

_High-level takeaways that apply beyond the competition._

1. 2 GPUs got to val_bpb 1.30 in 10 min — 8 GPUs should process ~4× more tokens in the same window, likely pushing below 1.22
2. Model still improving at wall clock cutoff — not converged, more throughput = better score
3. int8+zlib compression is essentially free (1.3033 → 1.3045, only +0.001 BPB degradation)
4. SP8192 dataset is NOT in the official PG repo (`willdepueoai/parameter-golf`). It's hosted on Kevin Clark's fork: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80`. All top 5 submissions (ranks 1-5) use this source.
5. ~~**SLM optimal ratio is k=0.8 at 17M scale**~~ **INVALID** — SLM code was not present during Session 6 runs. SLM validation pending (Session 7).
6. **Techniques stack cleanly** — SP8192 + TTT + LeakyReLU² + headwise + QKG5 all combine without interference. Best 2×H100: 1.2411 BPB (Run A). *(SLM stacking claim retracted pending validation.)*
7. **3-seed reproducibility confirmed** — SP8192 combo slim + TTT on 8×H100 gives mean 1.2073 BPB (std ±0.0006). Results are stable across random seeds.
8. **Total cost: ~$240+ across 30+ experiments** — systematic ablation approach validated each technique individually before stacking.

## On Metric Choice & Goodhart's Law

We use val_bpb (bits per byte on FineWeb validation) as our sole optimization target, consistent with the Parameter Golf competition metric. We acknowledge this is subject to Goodhart's Law — when a measure becomes a target, it ceases to be a good measure.

To keep our analysis honest, we categorize techniques by _what kind_ of improvement they provide:

| Category | What it means | Example techniques |
|----------|---------------|-------------------|
| **Architecture/training** | Genuinely better model — learns more per FLOP | Rho-1 token filtering, structured FFN, better LR schedules |
| **Throughput** | More steps in same wall clock — real improvement via more training | MQA speed gains, smaller batch accumulation |
| **Compression** | Smaller artifact, not better learning | int8→int6 quantization, layer tying for size |

Not all BPB improvements are equal. A technique that lowers BPB by fitting FineWeb's specific distribution (e.g., curriculum that mirrors the val set) is less valuable than one that improves the model's general language capability (e.g., better optimizer scheduling).

**Why BPB is still a reasonable proxy:** The Chinchilla paper (Hoffmann et al., 2022) showed that held-out perplexity improvements from principled scaling _do_ transfer to downstream tasks. The FineWeb paper (Penedo et al., 2024) confirmed that data quality improvements lowering BPB also improve MMLU/ARC. BPB isn't meaningless — it's just not the full picture.

**Our framing:** We study which techniques give real improvements under fixed compute constraints, while noting that some optimizations may be metric-specific rather than transferable.

## Crossover with Course Project

_Findings from Parameter Golf that are relevant to the nanochat training pipeline project._

- The 10-min wall clock constraint makes GPU count the primary lever — scaling from 2→8 GPUs is the easiest win
- Compression (int8+zlib) preserves model quality almost perfectly — useful for deployment constraints in the nanochat pipeline too
