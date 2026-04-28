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

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Quant | Size | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------|------|---------|
| E1† | Elementwise dim=448 GQA | ~16.4M | — | **1.2338** | 2,644 | — | int8 | 16.67 MB | **No** |
| 13† | SP8192 combo slim + TTT (re-run) | 16.36M | 3.1990 | 1.2384 | 2,749 | 218ms | int8 | 15.09 MB | Yes |
| D† | SP8192 combo slim + TTT (re-run) | 16.36M | 3.2021 | 1.2396 | 2,652 | 226ms | int8 | 15.08 MB | Yes |
| A† | SP8192 combo slim + TTT | 16.36M | 3.2059 | 1.2411 | 2,572 | 233ms | int8 | 15.03 MB | Yes |
| H† | SP8192 combo slim (no TTT) | 16.36M | 3.2112 | 1.2432 | 2,541 | 236ms | int8 | 15.04 MB | Yes |
| E2† | Elementwise dim=416 GQA | ~14.6M | — | 1.2447 | 2,772 | — | int8 | 14.68 MB | Yes |
| E3† | MQA dim=448 headwise | ~14.3M | — | 1.2509 | 2,979 | — | int8 | 14.32 MB | Yes |
| R3§ | **ResFormer α=0.5 10L MHA** | 27.8M | 3.2383 | **1.2536** | 2,535 | — | GPTQ | 15.55 MB | Yes |
| R1§ | ResFormer α=0.1 10L MHA | 27.8M | 3.2405 | 1.2545 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| R4§ | ResFormer α=0.7 10L MHA | 27.8M | 3.2420 | 1.2551 | 2,503 | — | GPTQ | 15.56 MB | Yes |
| A2‡ | Elem dim=512 9L MHA | 25.4M | 3.2488 | 1.2577 | 2,712 | — | GPTQ | 14.24 MB | Yes |
| Q0§ | Elem 10L MHA (GPTQ baseline) | 27.8M | 3.2506 | 1.2579 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| R0§ | ResFormer α=0.0 10L MHA | 27.8M | 3.2506 | 1.2584 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| L3‡ | Elem dim=512 11L GQA | 27.3M | 3.2525 | 1.2591 | 2,484 | — | GPTQ | 15.27 MB | Yes |
| E4† | MQA + Elementwise dim=416 | ~14.0M | — | 1.2601 | 2,982 | — | int8 | 14.02 MB | Yes |
| 3 | Elementwise gated attn* | 19.42M | 2.1280 | 1.2602 | 3,129 | 192ms | int8 | 17.87 MB | **No** |
| 7 | LeakyReLU² | 17.06M | 2.1344 | 1.2641 | 3,673 | 163ms | int8 | 15.77 MB | Yes |
| 8 | LeakyReLU² + headwise* | 17.10M | 2.1345 | 1.2642 | 3,368 | 178ms | int8 | 15.77 MB | Yes |
| 6v2 | Baseline repeat | 17.06M | 2.1357 | 1.2649 | 3,661 | 164ms | int8 | 15.77 MB | Yes |
| 2 | Headwise gated attn* | 17.10M | 2.1366 | 1.2653 | 3,287 | 182ms | int8 | 15.75 MB | Yes |
| L2‡ | Elem dim=512 10L GQA | 25.2M | 3.2712 | 1.2664 | 2,682 | — | GPTQ | 14.11 MB | Yes |
| 6 | Baseline (GQA) | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | int8 | 15.75 MB | Yes |
| D2‡ | Elem dim=512 9L GQA | 23.1M | 3.2765 | 1.2684 | 2,868 | — | GPTQ | 12.94 MB | Yes |
| 12 | Baseline (PyTorch 2.6) | 17.06M | 2.1462 | 1.2711 | 3,087 | 194ms | int8 | 15.70 MB | Yes |
| 9 | Headwise + QK-Gain 5.0 | 17.10M | 2.1475 | 1.2719 | 2,861 | 210ms | int8 | 15.65 MB | Yes |
| 4 | MQA (1 KV head) | 17.65M | 2.1549 | 1.2761 | 3,370 | 178ms | int8 | 16.84 MB | **No** |
| D1‡ | Elem dim=448 9L GQA | 18.1M | 3.3216 | 1.2859 | 2,618 | — | GPTQ | 10.20 MB | Yes |
| ~~5~~ | ~~INVALID (stale env)~~ | — | — | — | — | — | — | — | — |

**Runs 2-9, 12: SP1024, 10-min wall clock. Runs 2-9: PyTorch 2.11. Run 12: PyTorch 2.6 (18% slower per step). † Runs A, D, H, 13: SP8192, 2×H100, 2026-04-26. † Runs E1-E4: SP8192, 2×H100, 2026-04-27 (elementwise + MQA sweep). ‡ Runs D1-D4, L2, L3, A2: SP8192, 2×H100, GPTQ int7 + train data, 2026-04-28 (benchmark sweep). § Runs Q0, R0-R4: SP8192, 2×H100, GPTQ int7 + train data, 2026-04-28 (GPTQ tuning + ResFormer). GPTQ val_bpb = TTT BPB (post-quant + TTT recovery).**

**Runs D and 13** originally claimed SLM but SLM code was absent on the pod. They are additional Run A repeats (SP8192 combo slim + TTT, no SLM). Run-to-run variance: A=1.2411, D=1.2396, 13=1.2384 (spread 0.0027, consistent with noise).

> **Note:** Run 1 (2026-04-16) excluded — ran on 1×GPU (old pod, PyTorch 2.4.1), completed only 1,819 steps. Result (1.3045 BPB) is not comparable. Run 12 was an accidental vanilla baseline (config source failed, intended as SP8192 combo slim).

*\* Original technique by James Vo — gated attention applied post-SDPA with sigmoid gates, inspired by NeurIPS 2025 Best Paper (arxiv.org/abs/2505.06708).*

### Session 6 — Technique Stacking (SP1024, 1-GPU†)

> **⚠️ NGPUS bug:** These SP1024 runs used 1 GPU due to an env var overwrite bug (step_avg ~220ms vs normal ~163ms on 2×H100). Absolute BPB is inflated but **relative comparisons are valid.**

**Technique Stacking** (SP1024, 1-GPU — SLM labels removed, code was not present†):

| Technique | val_bpb | Steps | Notes |
|-----------|---------|-------|-------|
| LeakyReLU² + headwise | **1.2899** | 2,933 | Best SP1024 combo |
| LeakyReLU² | 1.2932 | 2,619 | |
| Headwise + QK-Gain 5.0 | 1.2981 | 2,668 | |

**Key findings:** (1) Techniques stack: LReLU²+headwise beats any single technique. (2) SP8192 still dominates: best SP1024 combo (1.2899 on 1-GPU) can't match SP8192 on same hardware (1.2411-1.2432).

† All runs 2026-04-26, PyTorch 2.6, 10-min wall clock. NGPUS bug confirmed by step_avg (~220ms vs expected ~163ms for 2×H100). SLM code was absent on this pod (commit `d7af1ec`, Apr 23, predates SLM push of Apr 26). Original "SLM sweep" and "+SLM" combo runs removed — all were running without SLM, BPB differences were run-to-run noise. See Session 7 for real SLM validation.

### Session 7 — SLM Validation (2×H100, SLM code confirmed present)

First real SLM runs with working code. Preflight check verified `slm_enabled` in `train_gpt.py` before running.

**SLM Ratio Sweep** (SP1024 GQA baseline + SLM, 2×H100):

| Run | Config | SLM k | val_bpb (int8) | vs Run 6v2 (1.2649) | Steps | Step avg |
|-----|--------|-------|----------------|---------------------|-------|----------|
| S1 | SP1024 GQA + SLM | 0.6 | 1.4204 | **+0.1555** | 3,318 | 181ms |
| S5 | SP1024 GQA + SLM | 0.95 | 1.2668 | +0.0019 | 3,403 | 176ms |
| 6v2 | SP1024 GQA (no SLM) | — | 1.2649 | — | 3,661 | 164ms |

**SLM on SP8192 competition config** (combo slim + TTT, 2×H100):

| Run | SLM k | val_bpb (TTT) | val_bpb (int8) | vs Run A (1.2411) | Steps | Step avg |
|-----|-------|---------------|----------------|-------------------|-------|----------|
| S2 | 0.6 | 1.4034 | 1.4002 | **+0.1592** | 2,654 | 226ms |
| S3 | 0.7 | 1.3201 | 1.3183 | +0.0790 | 2,677 | 224ms |
| S4 | 0.8 | — | 1.2652 | +0.0241 | 2,643 | 227ms |
| A | — | 1.2411 | 1.2420 | — | 2,572 | 233ms |

**Conclusion: SLM is harmful at 17M scale.** Every ratio tested (k=0.6 to k=0.95) produces worse BPB than the no-SLM baseline. The damage decreases as k approaches 1.0 (fewer tokens dropped), but never reaches parity. Even dropping just 5% of tokens (k=0.95) hurts by +0.0019 BPB.

**Why it fails — three compounding reasons:**

1. **Model too small.** Rho-1 showed gains on 1B+ params where the model has already learned common patterns (L→L tokens = 51% of data). At 17M params, the model is still learning "the", "of", "is" — skipping them removes gradient signal the model genuinely needs. The paper's assumption that easy tokens are wasted doesn't hold when the model hasn't mastered them yet.

2. **No reference model = can't distinguish learnable from unlearnable.** Our Option A (simple loss-threshold) keeps the top-k% tokens by raw loss. But high loss includes both H→L tokens (learnable — "Paris", "capital") AND H→H tokens (unlearnable noise — "Parisii", misspellings). The real Rho-1 uses a pre-trained reference model to compute *excess* loss (yours minus reference's), which filters out H→H tokens. Without this, we keep ~11% unlearnable noise while dropping useful medium-difficulty tokens.

3. **Wall clock budget too short.** With only 10 min of training (~3,000 steps on 2×H100), every step matters. SLM reduces effective batch size by (1-k)% — at k=0.6, each step trains on 40% fewer tokens. The paper's 10× convergence speedup claim was measured in total tokens processed, not wall-clock time. In our fixed-time regime, fewer tokens per step = fewer total tokens = worse model, period.

### Session 8 — Elementwise + MQA Sweep (2×H100, SP8192, 2026-04-27)

Testing whether elementwise gated attention and MQA can fit under 16 MB at reduced MODEL_DIM. Elementwise gave the best per-step BPB (Run 3, 1.2602) but busted budget at dim=512 (17.87 MB). MQA (Run 4) was also over at dim=512 (16.84 MB).

**Sweep results** (SP8192 combo slim base, 2×H100):

| Run | Config | val_bpb | Size (int8+zlib) | Budget? | vs Run A (1.2411) |
|-----|--------|---------|-------------------|---------|-------------------|
| **E1** | Elementwise dim=448 GQA | **1.2338** | 16.67 MB | **No (+0.67 MB)** | **-0.0073** |
| E2 | Elementwise dim=416 GQA | 1.2447 | 14.68 MB | Yes | +0.0036 |
| E3 | MQA dim=448 headwise | 1.2509 | 14.32 MB | Yes | +0.0098 |
| E4 | MQA + Elementwise dim=416 | 1.2601 | 14.02 MB | Yes | +0.0190 |

**No run passes** the dual criteria of beating Run A (1.2411) AND fitting under 16 MB.

**Key findings:**

1. **E1 is the best 2×H100 BPB ever (1.2338)** — beats Run A by 0.0073. But 16.67 MB, over budget by 0.67 MB. Tantalizingly close.
2. **Elementwise quality collapses at dim=416** — E2 (1.2447) is worse than Run A (1.2411). The dim reduction from 448→416 costs more BPB than elementwise gains.
3. **MQA confirmed worse on SP8192** — E3 (1.2509) is 0.0098 behind Run A. Fewer KV heads = worse quality, consistent with SP1024 result (Run 4).
4. **Combos don't stack** — E4 (MQA + elementwise, 1.2601) is worst of all. MQA's quality loss overwhelms elementwise's gain at dim=416.
5. **dim=432 untested** — could thread the needle between E1 (over budget) and E2 (under quality). 432 is divisible by 8 heads.

**Leaderboard insight:** Top PG entries solve the size problem differently — they don't shrink MODEL_DIM. Instead they use **int6 quantization** (compressed size ~25% smaller than int8) and **depth recurrence** (loop layers 4-5 for more virtual depth with same param count). Both are architectural/compression improvements our pipeline doesn't have yet.

---

## Competition Context

**Challenge:** Train the best language model in ≤ 10 min on 8×H100 SXM. Artifact ≤ 16 MB (16,000,000 bytes, decimal — code + compressed model). Scored by FineWeb validation BPB (bits per byte, tokenizer-agnostic). New SOTA must beat previous by ≥ 0.005 nats at p < 0.01.

**Naive Baseline:** 1.2244 BPB — 9L, 512 dim, 1024 vocab, tied embeddings, 4 KV heads, int8+zlib.

### Session 9 — GPTQ Validation (2×H100, SP8192, 2026-04-27)

Ported full GPTQ from PG rank 9 (Marko Sisovic). Algorithm: Frantar et al., "GPTQ", ICLR 2023. AR self-gen calibration (64 × 2048, no training data at eval time).

**First GPTQ run** (headwise dim=448, 5-percentile clip search, int6, AR self-gen):

| Metric | Value | Expected |
|--------|-------|----------|
| Pre-quant BPB | 1.2388 | — |
| Post-GPTQ int6 BPB (roundtrip) | 1.3450 | ~1.25 |
| Post-GPTQ int6 BPB (TTT) | **1.2929** | ~1.24 |
| GPTQ gap (roundtrip) | **+0.1062** | +0.01 |
| GPTQ gap (after TTT) | **+0.0541** | +0.002 |
| Artifact size | **10.50 MB** | ~10 MB |
| GPTQ calibration time | 117.6s | ~30s |

**Size is excellent** (10.50 MB, 5.5 MB headroom). **BPB gap is 10× worse than expected.** Kevin Clark (rank 5) gets +0.012 gap; we get +0.106. TTT partially rescued it (1.3450 → 1.2929) but that's TTT compensating for bad quantization.

**Bugs found and fixed:**
1. `torch.inference_mode()` in Hessian collection created "inference tensors" that poisoned the Rotary cos/sin cache, crashing TTT. Fixed by adding `.clone()` to Rotary output. Root cause: `inference_mode` creates permanently tainted tensors unlike `no_grad`. Kevin Clark uses `no_grad` — switching to match.
2. The 5-percentile clip search (rank 9/10 approach) runs GPTQ 5× per matrix. Kevin Clark uses `k × std(row)` single pass — same Cholesky error compensation, 5× faster, and directly controls compressed size (his README has the mathematical proof).

**Decompressed Kevin Clark's rank 5 code** (LZMA + base85 encoded, 416 lines). Key differences from our rank 9 port:
- Uses `torch.no_grad()` not `inference_mode()` — avoids the Rotary crash entirely
- Uses `clip_range=63` (int7) with `k=12.85` — same compressed size as int6, less clipping error
- Uses training data calibration (64 batches, ~5s) not AR self-gen (~120s)
- Uses PyTorch `register_forward_hook` for Hessian collection instead of manual `_save_gptq` flags
- Single GPTQ pass per matrix (k×std clip), not 5 passes (percentile search)

**Next: GPTQ benchmark sweep** — testing 4 combinations (int6/int7 × AR/train) with the updated code (no_grad + k×std clipping). See `runs/run_gptq_benchmark_2gpu.sh`.

**GPTQ benchmark results** (2×H100, headwise dim=448, SP8192, v2 code: no_grad + k×std clip):

| Run | Config | Pre-Q BPB | RT BPB | TTT BPB | Gap (vs Run A) | Size | GPTQ time |
|-----|--------|-----------|--------|---------|----------------|------|-----------|
| G1 | int7 + AR self-gen | 1.2361 | 1.3653 | 1.2930 | +0.0519 | 9.22 MB | 119s |
| **G2** | **int7 + train data** | **1.2364** | **1.3608** | **1.2924** | **+0.0513** | **9.22 MB** | **4s** |
| G3 | int6 + AR self-gen | 1.2364 | 1.3813 | 1.3081 | +0.0670 | 7.63 MB | 116s |
| G4 | int6 + train data | 1.2360 | 1.3801 | 1.3030 | +0.0619 | 7.63 MB | 4s |

**Decision: int7 + train data (G2 config).** Strictly dominates: same size as G1, better quality, 30× faster. int7 beats int6 by ~0.01 BPB gap at only +1.6 MB size cost.

**Key findings:**
1. **v2 code barely improved over v1** — best gap +0.0513 (G2) vs +0.0518 (v1). The no_grad and k×std fixes solved the Rotary crash and speed, but **not the quality gap**.
2. **Train data calibration > AR self-gen** — consistently better quality (G2 vs G1, G4 vs G3) and 30× faster (4s vs 116-119s).
3. **int7 > int6** — ~0.01 BPB less degradation at +1.6 MB size cost. At 9.22 MB, still 6.78 MB under budget.
4. **Gap still 4× worse than Kevin Clark** — +0.051 vs +0.012. Remaining difference likely in Hessian collection method (Kevin uses `register_forward_hook` vs our manual `_save_gptq` flags) or other subtle implementation details.

**Efficiency analysis (size saved vs BPB lost, relative to Run A int8+zlib):**

| Config | Size saved | BPB lost | Ratio (higher = better) |
|--------|-----------|----------|------------------------|
| G2 (int7, best) | 38.7% | 4.1% | 9.4× |
| G4 (int6, smallest) | 49.2% | 5.0% | 9.9× |
| Kevin Clark target | ~35% | ~1.0% | 35× |

Ratios are favorable (well above 1×), but the net tradeoff is negative for model upgrades: GPTQ frees ~5.8 MB for dim=512, but dim=512 only recovers -0.020 BPB while GPTQ costs +0.051. Need to close the gap to Kevin Clark's +0.012 before GPTQ becomes net-positive for bigger models.

### Session 10 — Benchmark Sweep: Dim / Layers / Attention (2×H100, GPTQ, 2026-04-28)

Purpose: "thicken" the model to fill GPTQ budget after compression. GPTQ shrinks ~30-50%, so we can afford bigger models. Three isolated sweeps, each varying one axis. All runs: elementwise gated attn + GPTQ int7 (clip=63) + train data calib + Score-First TTT.

**Sweep results** (2×H100, 10-min wall clock):

| Run | Sweep | Config | Params | Steps | Pre-Q BPB | TTT BPB | GPTQ Gap | Size | Under 16MB? |
|-----|-------|--------|--------|-------|-----------|---------|----------|------|-------------|
| D1 | dim | dim=448, 9L, GQA | 18.1M | 2,618 | 1.2322 | 1.2859 | +0.0537 | 10.20 MB | Yes |
| **D2** | **dim** | **dim=512, 9L, GQA** | **23.1M** | **2,868** | **1.2120** | **1.2684** | **+0.0564** | **12.94 MB** | **Yes** |
| D3 | dim | dim=768, 9L, GQA | 48.8M | 1,843 | 1.1856 | 1.2287 | +0.0431 | 27.14 MB | No |
| D4 | dim | dim=1024, 9L, GQA | 83.9M | 1,323 | 1.1869 | 1.2127 | +0.0258 | 46.47 MB | No |
| L2 | layer | dim=512, 10L, GQA | 25.2M | 2,682 | 1.2083 | 1.2664 | +0.0581 | 14.11 MB | Yes |
| L3 | layer | dim=512, 11L, GQA | 27.3M | 2,484 | 1.2042 | 1.2591 | +0.0549 | 15.27 MB | Yes |
| A2 | attn | dim=512, 9L, MHA | 25.4M | 2,712 | 1.2045 | 1.2577 | +0.0532 | 14.24 MB | Yes |

D2 is shared baseline for all 3 sweeps. PG baseline: 1.2244 BPB.

**Projected BPB if GPTQ gap matched Kevin Clark's (~0.012):**

| Run | Pre-Q BPB | Projected TTT BPB | Size | Beats baseline? |
|-----|-----------|-------------------|------|-----------------|
| D2 | 1.2120 | ~1.224 | 12.94 MB | Barely (at the line) |
| L2 | 1.2083 | ~1.220 | 14.11 MB | Yes |
| **L3** | **1.2042** | **~1.216** | **15.27 MB** | **Yes** |
| **A2** | **1.2045** | **~1.217** | **14.24 MB** | **Yes** |

**Key findings:**

1. **Pre-quant, everything beats baseline.** Even D2 (1.2120) crushes 1.2244. The model quality is there — compression is the sole bottleneck.
2. **GPTQ gap ~0.05 BPB is the bottleneck** — 4× worse than Kevin Clark's ~0.012. This single issue holds everything back.
3. **Bigger models have smaller GPTQ gaps.** D4 (1024) gap is only +0.0258 vs D1 (448) at +0.0537. More parameters = more redundancy for GPTQ to exploit.
4. **Best under-budget candidates** (if GPTQ gap is fixed to ~0.012): **L3** (11L, 1.2042 pre-Q, 15.27 MB) and **A2** (MHA, 1.2045 pre-Q, 14.24 MB). Both have ~1.204 pre-Q BPB with budget headroom.
5. **MHA beats GQA by -0.0107 TTT BPB** at +1.3 MB cost (A2 vs D2). Full KV heads improve quality when budget allows.

### Session 11 — GPTQ Tuning + ResFormer (2×H100, GPTQ, 2026-04-28)

Two experiments: (A) GPTQ quality improvements to close the 0.05 gap, (B) ResFormer value residual learning. Base config: dim=512, 10L, MHA, elementwise + GPTQ int7 + train data.

**Part A: GPTQ Tuning** (vs Q0 baseline: pre-Q 1.2035, TTT 1.2579, gap +0.0544):

| Run | Config | Pre-Q BPB | TTT BPB | GPTQ Gap | Size | GPTQ Time |
|-----|--------|-----------|---------|----------|------|-----------|
| Q0 | Baseline (10L MHA) | 1.2035 | 1.2579 | +0.0544 | 15.55 MB | 4s |
| Q1 | Sequential blocks | 1.2037 | 1.3916 | +0.1879 | 15.55 MB | 35s |
| Q3 | GPTQ on embeddings | 1.2025 | 1.6897 | +0.4872 | 15.36 MB | 31s |
| Q7 | All combined | 1.2039 | 1.8679 | +0.6640 | 15.36 MB | 63s |

**All GPTQ tuning runs made things dramatically worse.** Sequential (+0.19), embed GPTQ (+0.49), all combined (+0.66). Pre-quant BPB is unaffected (GPTQ runs post-training), confirming the damage is entirely in the quantization step. Root causes:
1. **Sequential block quantization** — replacing weights with dequantized versions introduces cumulative error. Even with the fix (save/restore original weights, use sequential Hessians only), the Hessians collected through dequantized blocks are worse than full-precision Hessians.
2. **Embedding GPTQ** — frequency-weighted column correlation `H = W^T @ diag(freq) @ W` is not the right Hessian for embedding quantization. Embeddings are lookup tables, not linear projections — GPTQ's column-by-column error compensation doesn't apply correctly.
3. **Combined** — errors compound when both are enabled.

**Conclusion: abandon GPTQ tuning approaches.** The existing GPTQ (Q0 config, gap +0.054) is our best. The gap to Kevin Clark likely comes from his `register_forward_hook` implementation details or his looped-layer architecture, not from sequential quantization or embedding GPTQ.

**Why leaderboard GPTQ gaps are 4-5× smaller than ours** (analysis of Kevin Clark rank 5, dexhunter rank 7 READMEs):

| Who | Pre-Q BPB | Post-Q BPB | GPTQ Gap | Notes |
|-----|-----------|------------|----------|-------|
| Kevin Clark (rank 5) | 1.090 | 1.102 | **+0.012** | Full quantization-aware stack |
| dexhunter (rank 7) | 1.099 | 1.109 | **+0.010** | All-int6, WD=0.09, EMA |
| Us (R3, best GPTQ) | 1.200 | 1.254 | **+0.053** | Bare GPTQ only |

Five compounding reasons the leaderboard achieves ~0.01 gap vs our ~0.05:

1. **Depth recurrence = fewer unique matrices to quantize.** Kevin Clark loops layers 4-5 (sharing weights), so ~8 unique layer sets instead of 10. Fewer matrices = less total quantization error. This is the most elegant insight: you can improve quantization quality by reducing the number of surfaces GPTQ must compress.
2. **QAT (Quantization-Aware Training).** Ranks 7-12 use soft-round QAT — the model trains expecting quantization. Our model trains full-precision then gets shocked post-hoc.
3. **Higher weight decay (0.085-0.090).** dexhunter's key finding: "higher WD produces smaller weights that compress 5% better under brotli." Smaller weights = less dynamic range = less GPTQ error.
4. **EMA (Exponential Moving Average, decay ~0.997).** Ranks 7-14 all use EMA. Averaging out training noise makes weights smoother and more compressible.
5. **`register_forward_hook` Hessian collection.** Kevin Clark captures true activation statistics through the live network; our manual `_save_gptq` flags likely miss some dynamics.

**Key takeaway:** The leaderboard doesn't just "use GPTQ" — they use GPTQ as the final step of a quantization-aware pipeline (WD tuning → EMA → QAT → GPTQ → brotli). We're only doing the last two steps. Closing the gap requires adopting the full pipeline, not tuning GPTQ parameters.

**Part B: ResFormer Alpha Sweep** (vs R0 control: pre-Q 1.2040, TTT 1.2584):

| Run | Alpha | Pre-Q BPB | TTT BPB | GPTQ Gap | Size |
|-----|-------|-----------|---------|----------|------|
| R0 | 0.0 | 1.2040 | 1.2584 | +0.0544 | 15.55 MB |
| R1 | 0.1 | 1.2020 | 1.2545 | +0.0525 | 15.55 MB |
| R3 | 0.5 | **1.2004** | **1.2536** | +0.0532 | 15.55 MB |
| R4 | 0.7 | 1.2025 | 1.2551 | +0.0526 | 15.56 MB |

Note: R2 (alpha=0.3) not run. R3 is alpha=0.5 (run ID `resformer_a05`).

**ResFormer works.** Best results at alpha=0.5:
- Pre-Q BPB: **1.2004** (vs 1.2040 control, -0.0036 improvement)
- TTT BPB: **1.2536** (vs 1.2584 control, -0.0048 improvement)
- GPTQ gap slightly improved: +0.0532 vs +0.0544
- Zero extra params, zero size cost (15.55 MB unchanged)

**Key findings:**
1. **Alpha=0.5 is optimal** — equal blend of V₀ and V_current gives best pre-Q and TTT BPB
2. **Diminishing returns past 0.5** — alpha=0.7 is worse than 0.5, suggesting too much V₀ drowns out layer-specific value information
3. **GPTQ gap also improved** — 0.0532 vs 0.0544, suggesting V₀ residual makes weights more compressible (smoother value distribution)
4. **Free improvement** — no extra params, no extra memory, no extra compute, no size increase
5. **Projected 8×H100** with alpha=0.5: pre-Q ~1.168 (from 2×H100 scaling factor), TTT ~1.221 — would beat baseline (1.2244) even after GPTQ

---

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

### Run D — SP8192 Combo Slim + TTT (Run A repeat), 2×H100 (2026-04-26)

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 |
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
- Run A repeat. BPB difference vs Run A (1.2411 → 1.2396) is run-to-run noise (±0.0015)

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

### Run 13 — SP8192 Combo Slim + TTT (Run A repeat), 2×H100 (2026-04-26)

| Item | Value |
|---|---|
| Date | 2026-04-26 |
| GPUs | 2× H100 |
| Technique | SP8192 + Score-First TTT + LeakyReLU² + QK-Gain 5.0 + Headwise + MODEL_DIM=448 |
| Params | 16,364,616 (~16.4M) |
| Vocab | 8192 (SentencePiece BPE) |
| Steps completed | 2,749 / 20,000 (hit 10-min wall clock cap) |
| Step avg | 218 ms |
| **val_bpb (TTT)** | **1.2384** |
| Model size (int8+zlib) | **15.09 MB** (under 16 MB budget) |
| PyTorch version | 2.6.0 |

**Observations:**
- Run A repeat. BPB variance across A/D/13: 1.2411, 1.2396, 1.2384 (spread ±0.0014, normal noise)
- Step count variance: A=2,572, D=2,652, 13=2,749 (spread ±89 steps, normal)

### Runs A–H–D–13 — SP8192 2×H100 Ablation Summary

| Run | TTT | val_bpb | TTT Δ | Notes |
|-----|-----|---------|-------|-------|
| H | No | 1.2432 | — | No-TTT baseline |
| A | Yes | 1.2411 | -0.0021 | TTT adds ~0.002 BPB |
| D | Yes | 1.2396 | -0.0021 | Run A repeat (noise) |
| 13 | Yes | 1.2384 | -0.0021 | Run A repeat (noise) |

## Techniques That Worked

_Add entries as we discover things._

| Technique | Impact on BPB | Why it works |
|---|---|---|
| **SP8192 vocab scaling** | 1.2411 (2×H100) vs 1.2649 SP1024 = **-0.024 BPB** | Bigger vocab = fewer tokens per byte = lower BPB. Single biggest lever. All top 8 leaderboard entries use SP4096/SP8192. Paper: "Scaling Laws with Vocabulary" (NeurIPS 2024). |
| **Score-First TTT** | -0.0021 BPB (1.2432→1.2411) | Eval-time fine-tuning: adapt model to validation distribution. Legal under PG rules (score before update). Source: @dexhunter (PG competition). |
| Gated Attention (headwise) | 1.2653 (compressed), fits 16 MB | Sigmoid gate after SDPA lets model suppress uninformative heads per token. Nearly free: +37K params, no speed penalty. Original technique by James Vo. |
| LeakyReLU² | -0.0008 BPB vs baseline (1.2641 vs 1.2649) | Free activation swap — no extra params, no speed cost. Used by PG ranks 10-11. |

## Techniques That Didn't Work

| Technique | Expected impact | Actual result | Why it failed |
|---|---|---|---|
| Gated Attention (elementwise) | Better BPB than headwise | Best BPB (1.2338 at dim=448) but 16.67 MB over budget; dim=416 fits but 1.2447 worse than Run A | Elementwise needs dim≥448 to beat headwise, but that's over 16 MB. Shrinking dim kills the gain. No sweet spot found. |
| MQA on SP8192 | Faster inference, smaller model | 1.2509 BPB at dim=448 — 0.0098 worse than GQA (Run A) | Confirmed on SP8192 (Session 8) after SP1024 (Run 4). Fewer KV heads = worse quality at 17M scale. |
| QK-Gain 5.0 (on SP1024) | Better attention scaling | 1.2719 BPB (worse than headwise 1.2653) | 15% slower steps (210ms vs 182ms), higher VRAM (13GB vs 10GB). QK-Gain 5.0 likely needs SP8192 to be effective. |
| SLM / Rho-1 (all ratios) | Better per-step learning by filtering easy tokens | k=0.6: +0.155 BPB, k=0.8: +0.024, k=0.95: +0.002 — ALL worse than no-SLM | At 17M params, model needs every gradient signal. Rho-1 paper tested at 1B+; doesn't transfer down. Simple loss-threshold (Option A) too crude without reference model. Paper: "Not All Tokens Are What You Need" (NeurIPS 2024). |

## Key Insights

_High-level takeaways that apply beyond the competition._

1. 2 GPUs got to val_bpb 1.30 in 10 min — 8 GPUs should process ~4× more tokens in the same window, likely pushing below 1.22
2. Model still improving at wall clock cutoff — not converged, more throughput = better score
3. int8+zlib compression is essentially free (1.3033 → 1.3045, only +0.001 BPB degradation)
4. SP8192 dataset is NOT in the official PG repo (`willdepueoai/parameter-golf`). It's hosted on Kevin Clark's fork: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80`. All top 5 submissions (ranks 1-5) use this source.
5. **SLM (Rho-1) doesn't work at 17M scale** — validated in Session 7 with working code. Every ratio (k=0.6 to k=0.95) hurts. Small models need all tokens; the paper's 1B+ results don't transfer down.
6. **Techniques stack cleanly** — SP8192 + TTT + LeakyReLU² + headwise + QKG5 all combine without interference. Best 2×H100: 1.2411 BPB (Run A).
7. **3-seed reproducibility confirmed** — SP8192 combo slim + TTT on 8×H100 gives mean 1.2073 BPB (std ±0.0006). Results are stable across random seeds.
8. **Total cost: ~$240+ across 30+ experiments** — systematic ablation approach validated each technique individually before stacking.
9. **Elementwise gated attention: best BPB but no budget-legal sweet spot** — E1 (dim=448, 1.2338) is the best 2×H100 BPB ever but 0.67 MB over. dim=416 fits but loses all quality gain. MQA also confirmed worse on SP8192.
10. **Next frontier: int6 quantization and depth recurrence** — every top-9 PG entry uses depth recurrence (loop layers 4-5), and most use int6 GPTQ instead of int8+zlib. These would let us keep dim=512 (or elementwise at dim=448) under 16 MB.

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
