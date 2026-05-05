# Parameter Golf — Findings & Insights

> What we learned from participating. Insights that could feed back into the course project.

## Official Runs — 8×H100 (sorted by BPB, best first)

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Quant | Size | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------|------|---------|
| **P4b●** | **PR #1851 + headwise + SOTA hparams + CaseOps (seed 1337)** | **35.99M** | **2.3242** | **1.0621** | **4,948** | — | **int6+brotli** | **15.98 MB** | **Yes** |
| ~~P3●~~ | ~~PR #1851 fork + headwise + EMA=0.990 + small batch + emb6 (3-seed mean)~~ | ~~35.99M~~ | — | ~~1.0066~~ | ~~12,382~~ | — | ~~int6+brotli~~ | ~~~15.97 MB~~ | ~~Yes~~ |
| P3-fix | P3 rerun with CASEOPS_ENABLED=1 (correct byte accounting, seed 42) | 35.99M | 2.4012 | 1.0972 | 10,796 | — | int6+brotli | 15.98 MB | Yes |
| *frontier* | *ndokutovich #1967 (N-gram Tilt + LeakyReLU 0.3)* | *~36M* | — | *1.0585* | — | — | — | — | *ref* |
| *SOTA* | *codemath3000 #1855 (SmearGate+LQER+9HP)* | *~36M* | — | *1.0611* | *~4,930* | — | *int6+lrzip* | *~15.90 MB* | *ref* |
| P1a★ | V2 C6 + SOTA hparams (6 overrides) | 35.99M | 2.7816 | 1.0769 | 4,587 | — | int6+brotli | 16.35 MB | **No** |
| P1b★ | V2 C6 + SOTA hparams + NUM_LOOPS=3 | 35.99M | 2.7881 | 1.0793 | 4,126 | — | int6+brotli | 16.36 MB | **No** |
| **C6●** | **V2 Headwise + emb7+eclip15 (3-seed mean)** | **35.99M** | **2.7910** | **1.0805** | **4,467** | — | **int6+brotli** | **15.70 MB** | **Yes** |
| L1◆ | V2 C6 + EMA=0.990 (legal) | 35.99M | 2.7976 | 1.0830 | 4,486 | — | int6+brotli | 15.75 MB | Yes |
| A1● | V2 F1 control (no additions) | 35.94M | 2.7912 | 1.0806 | 4,580 | — | int6+brotli | 15.98 MB | Yes |
| L2◆ | V2 C6 + Small Batch + EMA=0.990 | 35.99M | 2.8224 | 1.0926 | 13,146 | — | int6+brotli | 15.74 MB | Yes |
| A3● | V2 F2 headwise (default compression) | 35.99M | 2.7899 | 1.0801 | — | — | int6+brotli | 15.99 MB | Tight |
| A2● | V2 F7 (PR+RF, α=0.5) | 35.94M | 2.7971 | 1.0828 | 4,516 | — | int6+brotli | 15.98 MB | Yes |
| 10 | SP8192 combo + TTT | 20.77M | 3.0666 | 1.1872 | 10,582 | 57ms | int8+zlib | 19.41 MB | **No** |
| 11 | SP8192 combo slim + TTT | 16.36M | 3.1197 | 1.2077 | 11,073 | 54ms | int8+zlib | 15.35 MB | Yes |

**V2 runs: PyTorch 2.11, CUDA 13.0, FA3, SP8192, 10-min wall clock. Runs 10-11: PyTorch 2.6. PG baseline: 1.2244 BPB. SOTA: 1.0611 (codemath3000, PR #1855).**

**P4b is NEW BEST at 1.0621 BPB** (CaseOps sidecar, SOTA hparams, headwise gate, CLIP=13.0). Matches external SOTA (1.0611, delta 0.001). Under budget at 15.98 MB. Seed 1337. Byte ratio 0.3166 confirmed correct. P3 retracted (inflated byte denominator). C6 (1.0805) was previous best.

**◆ L1: EMA=0.990 HURT on 8×H100** (+0.0025 vs C6). With only ~4,486 steps, aggressive EMA averages too few checkpoints. **L2: Small Batch + EMA=0.990 HURT EVEN MORE** (+0.0121 vs C6). Despite 13,146 steps (where EMA=0.990 helped on 2×H100), the smaller batch size degrades quality more than extra steps help. EMA tuning does not transfer from 2×H100 to 8×H100 at any batch size.

**★ P1a: SOTA hparams improve BPB by −0.0036 vs C6** (1.0769 vs 1.0805) but over budget (16.35 MB). 6 overrides from PR #1855: WARMDOWN=0.85, MIN_LR=0.10, MATRIX_CLIP_SIGMAS=11.5, EMBED_CLIP_SIGMAS=14.0, BETA2=0.99, GPTQ_RESERVE=0.5. Looser MATRIX_CLIP_SIGMAS (11.5 vs 12.85) is the likely budget-buster. **P1b: NUM_LOOPS=3 hurts** — fewer steps (4,126 vs 4,587), throughput penalty > depth benefit (+0.0024 worse than P1a). Keep NUM_LOOPS=2.

> **PreQuantTTT ruled C3 violation** (score-after-adapt). okezue withdrew PR #1958 for the same issue — training on val data before the reported eval. R4 and X1 results below used PreQuantTTT and are **non-compliant**. PPM byte mixtures also ruled invalid (C2 violation, PR #1905 — probability distribution doesn't sum to 1). **P3 retracted** — byte accounting error (inflated denominator from CaseOps LUT). Corrected P3 = 1.0972 BPB (worse than C6). **Legal best: P4b at 1.0621 BPB.** External SOTA: 1.0611 (codemath3000 PR #1855).

### 3-Seed Reproducibility — V2 C6 (Headwise + emb7+eclip15), 8×H100

| Seed | val_loss (TTT) | val_bpb (TTT) | Steps | Weights (int6+brotli) | Eval time |
|------|----------------|---------------|-------|-----------------------|-----------|
| 42 | 2.7945 | 1.0818 | 4,469 | 15,697,552 (15.70 MB) | 394s |
| 1337 | 2.7883 | **1.0794** | 4,465 | 15,694,065 (15.69 MB) | 335s |
| 2025 | 2.7908 | 1.0804 | 4,467 | 15,693,855 (15.69 MB) | 334s |
| **Mean** | **2.7912** | **1.0805** | **4,467** | — | — |
| **Std** | — | **±0.0012** | — | — | — |

**Confirms robustness:** 3 seeds produce consistent results (std 0.0012 BPB). All under 16 MB budget. Train time ~588s, eval (sliding + TTT) 334-394s — both within 600s limits.

### 3-Seed Reproducibility — P3 (PR #1851 fork + headwise + EMA=0.990 + small batch + emb6), 8×H100

| Seed | Pre-Q BPB | Quant BPB | TTT BPB | Size (bytes) | Budget? |
|------|-----------|-----------|---------|--------------|---------|
| 42   | 1.0025    | 1.0205    | 1.0069  | 15,975,827   | Yes |
| 1337 | 1.0017    | 1.0190    | **1.0057** | 15,973,108 | Yes |
| 2025 | 1.0030    | 1.0206    | 1.0073  | 15,973,714   | Yes |
| **Mean** | **1.0024** | **1.0200** | **1.0066** | **~15,974K** | **Yes** |
| **Std** | **0.0007** | **0.0009** | **±0.0009** | — | — |

**RETRACTED — byte accounting error.** These BPB values used the inflated CaseOps LUT byte denominator (~164.6M) instead of canonical sidecar bytes (~151M). Rerun with `CASEOPS_ENABLED=1` gives seed 42 TTT BPB = **1.0972** (worse than C6 1.0805). The val_loss is real (~2.401) but the BPB conversion was wrong. See "PR #2071 Legality Feedback" section below.

### 8×H100 Ablation — Technique Contributions (seed 42)

| Run | Config | TTT BPB | Weights | vs A1 (control) |
|-----|--------|---------|---------|-----------------|
| **A3** | **F2 headwise (default compression)** | **1.0801** | **15,993,169** | **-0.0005 (better)** |
| A1 | F1 (rank 1 default, no additions) | 1.0806 | 15,977,755 | — |
| S1 | C6 (headwise + emb7+eclip15) | 1.0818 | 15,697,552 | +0.0012 (worse) |
| A2 | F7 (PR+RF, α=0.5) | 1.0828 | 15,983,964 | +0.0022 (worse) |

**Headwise gate helps (-0.0005 BPB)** — A3 (F2 headwise, 1.0801) beats A1 (control, 1.0806). The effect is small but consistent with the 2×H100 result (F2=1.1636 vs F1=1.1641). However, **compression tuning costs +0.0017 BPB** — C6 (1.0818) is worse than A3 (1.0801) due to emb7+eclip15. The tighter embedding quantization (int7 + clip 15.0) hurts quality more than it saves in size. **ResFormer (α=0.5) hurts** — A2 (1.0828) is the worst of all four.

**A3 is over budget** — 15,993,169 bytes (15.99 MB), tight but technically under 16,000,000. Total with code: 16,043,196 bytes — **over 16 MB total**. Same issue as F2 on 2×H100 (16,007,049 bytes). The headwise gate adds ~16K bytes that barely bust the budget.

**2×H100 → 8×H100 scaling:** all V2 configs improved dramatically with 4× more GPUs. F1: 1.1641 → 1.0806 (−0.0835). F2: 1.1636 → 1.0801 (−0.0835). C6: 1.1622 → 1.0818 (−0.0804). Technique deltas preserved at scale.

### 3-Seed Reproducibility — SP8192 Combo Slim + TTT, 8×H100 (V1, historical)

| Seed | val_loss (TTT) | val_bpb (TTT) | Steps | Size (int8+zlib) |
|------|----------------|---------------|-------|-------------------|
| 42 | 3.1200 | 1.2078 | 11,028 | 15.34 MB |
| 1337 | 3.1169 | **1.2067** | 11,030 | 15.34 MB |
| 2025 | 3.1190 | 1.2075 | 11,036 | 15.34 MB |
| **Mean** | **3.1186** | **1.2073** | **11,031** | — |
| **Std** | — | **±0.0006** | — | — |

*V1 stack (our original code). Kept for historical reference — V2 results above supersede these.*

## Experiment Runs — 2×H100 (sorted by BPB, best first)

| Run | Technique | Params | val_loss | val_bpb | Steps | Step avg | Quant | Size | Budget? |
|-----|-----------|--------|----------|---------|-------|----------|-------|------|---------|
| **N1◆** | **V2 C6 + EMA=0.990 + Small Batch** | **35.99M** | **2.9365** | **1.1368** | **4,221** | — | **int6+brotli** | **15.70 MB** | **Yes** |
| S1▼ | V2 N1 + Cross-Seq Attn (eval-only, Paper #29) | 35.99M | 2.9397 | 1.1380† | 4,219 | — | int6+brotli | 15.71 MB | Yes |
| **B2▲** | **V2 C6 + Small Batch ga=1 (Paper #15) (James-experiment-2)** | **35.99M** | **2.9512** | **1.1419** | **3,349** | — | **int6+brotli** | **15.71 MB** | **Yes** |
| B3▲ | V2 C6 + Small Batch ga=1 + beta2=0.99 (James-experiment-2) | 35.99M | 2.9503 | 1.1422 | 3,349 | — | int6+brotli | 15.71 MB | Yes |
| R3◆ | V2 C6 + EMA=0.990 | 35.99M | — | 1.1505 | — | — | int6+brotli | 15.71 MB | Yes |
| N2◆ | V2 C6 + EMA=0.990 + Small Batch + DiffAttn (Paper #19) | 35.99M | 2.9722 | 1.1506 | 3,292 | — | int6+brotli | 15.71 MB | Yes |
| R2◆ | V2 C6 + EMA=0.993 | 35.99M | — | 1.1526 | — | — | int6+brotli | 15.71 MB | Yes |
| R1◆ | V2 C6 + EMA=0.995 + WD=0.10 | 35.99M | — | 1.1559 | — | — | int6+brotli | 15.71 MB | Yes |
| **E1●** | **V2 C6 + EMA=0.995** | **35.99M** | — | **1.1562** | — | — | **int6+brotli** | **15.71 MB** | **Yes** |
| P0★ | V2 C6 baseline (Session 15 control) | 35.99M | — | 1.1572 | — | — | int6+brotli | 15.71 MB | Yes |
| P1★ | V2 C6 + LR Warmup 2% | 35.99M | — | 1.1596 | — | — | int6+brotli | 15.70 MB | Yes |
| P2★ | V2 C6 + LR Warmup 5% | 35.99M | — | 1.1614 | — | — | int6+brotli | 15.70 MB | Yes |
| W2● | V2 C6 + WD=0.10 | 35.99M | — | 1.1619 | — | — | int6+brotli | 15.71 MB | Yes |
| **C6◇** | **V2 Headwise + emb7+eclip15** | **35.99M** | — | **1.1622** | — | — | **int6+brotli** | **15.71 MB** | **Yes** |
| P3★ | V2 C6 + LR Warmup 10% | 35.99M | — | 1.1638 | — | — | int6+brotli | 15.70 MB | Yes |
| F2¶ | V2 PR + Headwise Gate | 35.99M | — | 1.1636 | 1,030 | — | int6+brotli | 16.01 MB | Tight |
| F7¶ | V2 PR+RF + No Gate | 35.94M | — | 1.1636 | — | — | int6+brotli | 15.99 MB | Tight |
| F1¶ | V2 PR + No Gate (CTRL) | 35.94M | — | 1.1641 | 1,058 | — | int6+brotli | 15.99 MB | Tight |
| F8¶ | V2 PR+RF + Headwise Gate | 35.99M | — | 1.1650 | — | — | int6+brotli | 16.01 MB | Tight |
| C1◇ | V2 Headwise + emb7 | 35.99M | — | 1.1656 | — | — | int6+brotli | 15.48 MB | Yes |
| F5¶ | V2 RF + Headwise Gate | 35.99M | — | 1.1661 | 1,036 | — | int6+brotli | 16.01 MB | Tight |
| F4¶ | V2 RF + No Gate | 35.94M | — | 1.1666 | 1,044 | — | int6+brotli | 15.99 MB | Tight |
| C2◇ | V2 Headwise + emb6 | 35.99M | — | 1.1735 | — | — | int6+brotli | 14.97 MB | Yes |
| 13† | SP8192 combo slim + TTT (re-run) | 16.36M | 3.1990 | 1.2384 | 2,749 | 218ms | int8 | 15.09 MB | Yes |
| D† | SP8192 combo slim + TTT (re-run) | 16.36M | 3.2021 | 1.2396 | 2,652 | 226ms | int8 | 15.08 MB | Yes |
| A† | SP8192 combo slim + TTT | 16.36M | 3.2059 | 1.2411 | 2,572 | 233ms | int8 | 15.03 MB | Yes |
| H† | SP8192 combo slim (no TTT) | 16.36M | 3.2112 | 1.2432 | 2,541 | 236ms | int8 | 15.04 MB | Yes |
| E2† | Elementwise dim=416 GQA | ~14.6M | — | 1.2447 | 2,772 | — | int8 | 14.68 MB | Yes |
| E3† | MQA dim=448 headwise | ~14.3M | — | 1.2509 | 2,979 | — | int8 | 14.32 MB | Yes |
| R3§ | ResFormer α=0.5 10L MHA | 27.8M | 3.2383 | 1.2536 | 2,535 | — | GPTQ | 15.55 MB | Yes |
| R1§ | ResFormer α=0.1 10L MHA | 27.8M | 3.2405 | 1.2545 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| R4§ | ResFormer α=0.7 10L MHA | 27.8M | 3.2420 | 1.2551 | 2,503 | — | GPTQ | 15.56 MB | Yes |
| A2‡ | Elem dim=512 9L MHA | 25.4M | 3.2488 | 1.2577 | 2,712 | — | GPTQ | 14.24 MB | Yes |
| Q0§ | Elem 10L MHA (GPTQ baseline) | 27.8M | 3.2506 | 1.2579 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| R0§ | ResFormer α=0.0 10L MHA | 27.8M | 3.2506 | 1.2584 | 2,480 | — | GPTQ | 15.55 MB | Yes |
| L3‡ | Elem dim=512 11L GQA | 27.3M | 3.2525 | 1.2591 | 2,484 | — | GPTQ | 15.27 MB | Yes |
| E4† | MQA + Elementwise dim=416 | ~14.0M | — | 1.2601 | 2,982 | — | int8 | 14.02 MB | Yes |
| 7 | LeakyReLU² | 17.06M | 2.1344 | 1.2641 | 3,673 | 163ms | int8 | 15.77 MB | Yes |
| 8 | LeakyReLU² + headwise* | 17.10M | 2.1345 | 1.2642 | 3,368 | 178ms | int8 | 15.77 MB | Yes |
| 6v2 | Baseline repeat | 17.06M | 2.1357 | 1.2649 | 3,661 | 164ms | int8 | 15.77 MB | Yes |
| 2 | Headwise gated attn* | 17.10M | 2.1366 | 1.2653 | 3,287 | 182ms | int8 | 15.75 MB | Yes |
| L2‡ | Elem dim=512 10L GQA | 25.2M | 3.2712 | 1.2664 | 2,682 | — | GPTQ | 14.11 MB | Yes |
| 6 | Baseline (GQA) | 17.06M | 2.1388 | 1.2667 | 3,500 | 171ms | int8 | 15.75 MB | Yes |
| D2‡ | Elem dim=512 9L GQA | 23.1M | 3.2765 | 1.2684 | 2,868 | — | GPTQ | 12.94 MB | Yes |
| 12 | Baseline (PyTorch 2.6) | 17.06M | 2.1462 | 1.2711 | 3,087 | 194ms | int8 | 15.70 MB | Yes |
| 9 | Headwise + QK-Gain 5.0 | 17.10M | 2.1475 | 1.2719 | 2,861 | 210ms | int8 | 15.65 MB | Yes |
| P4★ | V2 C6 + Structured FFN r=0.5 b=4 | 23.0M | — | 1.1997 | — | — | int6+brotli | 13.90 MB | Yes |
| P5★ | V2 C6 + Structured FFN r=0.75 b=8 | 25.2M | — | 1.2068 | — | — | int6+brotli | 12.98 MB | Yes |
| D1‡ | Elem dim=448 9L GQA | 18.1M | 3.3216 | 1.2859 | 2,618 | — | GPTQ | 10.20 MB | Yes |

### Over-Budget Runs (reference only)

Runs that exceeded the 16 MB budget. Kept for BPB/technique comparison but not submission candidates.

| Run | Technique | Params | val_bpb | Quant | Size | Over by |
|-----|-----------|--------|---------|-------|------|---------|
| C8◇ | V2 Headwise + clip10+eclip15 | 35.99M | **1.1591** | int6+brotli | 17.54 MB | +1.54 MB |
| C7◇ | V2 Headwise + emb7+clip10+eclip15 | 35.99M | 1.1596 | int6+brotli | 17.01 MB | +1.01 MB |
| C4◇ | V2 Headwise + clip8 | 35.99M | 1.1598 | int6+brotli | 18.67 MB | +2.67 MB |
| C3◇ | V2 Headwise + clip10 | 35.99M | 1.1605 | int6+brotli | 17.31 MB | +1.31 MB |
| C5◇ | V2 Headwise + emb7+clip10 | 35.99M | 1.1620 | int6+brotli | 16.78 MB | +0.78 MB |
| F3¶ | V2 PR + Elementwise Gate | 38.83M | 1.1665 | int6+brotli | 17.21 MB | +1.21 MB |
| F9¶ | V2 PR+RF + Elementwise Gate | 38.83M | 1.1686 | int6+brotli | 17.22 MB | +1.22 MB |
| F6¶ | V2 RF + Elementwise Gate | 38.83M | 1.1700 | int6+brotli | 17.22 MB | +1.22 MB |
| E1† | Elementwise dim=448 GQA | ~16.4M | 1.2338 | int8 | 16.67 MB | +0.67 MB |
| 3 | Elementwise gated attn* | 19.42M | 1.2602 | int8 | 17.87 MB | +1.87 MB |
| 4 | MQA (1 KV head) | 17.65M | 1.2761 | int8 | 16.84 MB | +0.84 MB |
| ~~5~~ | ~~INVALID (stale env)~~ | — | — | — | — | — |

*Note: Lower MATRIX_CLIP_SIGMAS (C3/C4/C8) improves BPB but increases compressed size — tighter clipping changes value distribution in a way that compresses worse under brotli. C8 achieves best-ever BPB (1.1591) but at 17.54 MB.*

**Runs 2-9, 12: SP1024, 10-min wall clock. Runs 2-9: PyTorch 2.11. Run 12: PyTorch 2.6 (18% slower per step). † Runs A, D, H, 13: SP8192, 2×H100, 2026-04-26. † Runs E1-E4: SP8192, 2×H100, 2026-04-27 (elementwise + MQA sweep). ‡ Runs D1-D4, L2, L3, A2: SP8192, 2×H100, GPTQ int7 + train data, 2026-04-28 (benchmark sweep). § Runs Q0, R0-R4: SP8192, 2×H100, GPTQ int7 + train data, 2026-04-28 (GPTQ tuning + ResFormer). ¶ Runs F1-F9: V2 factorial (rank 1 fork + our techniques), SP8192, 2×H100, FA3, int6+brotli, 2026-04-28. ◇ Runs C1-C8: V2 compression tuning (F2 headwise base + compression knob variants), SP8192, 2×H100, FA3, int6+brotli, 2026-04-29. ● Runs C6/A1/A2 (8×H100): V2 C6 submission + ablation, SP8192, 8×H100, PyTorch 2.11+cu130, FA3, int6+brotli, 2026-04-29. All V2 runs: val_bpb = TTT BPB. Size = weights only (code adds 16.6-50 KB depending on LZMA compression).**

**★ Runs P0-P5: Paper #16 (LR Warmup) + Paper #5 (Structured FFN) A/B tests, SP8192, 2×H100, FA3, int6+brotli, 2026-04-30. C6 base config (headwise + emb7+eclip15). LR warmup: all 3 fractions hurt monotonically (more warmup = worse). Structured FFN: dramatic param/size savings (23-25M, 13-14 MB) but BPB degrades by +0.04-0.05. Both techniques FAIL on V2 stack.**

**★ Runs P1a-P1b (8×H100): SOTA hparam adoption Phase 1, SP8192, 8×H100, PyTorch 2.11+cu130, FA3, int6+brotli, 2026-04-30. 6 env-var overrides from PR #1855 (WARMDOWN=0.85, MIN_LR=0.10, MATRIX_CLIP_SIGMAS=11.5, EMBED_CLIP_SIGMAS=14.0, BETA2=0.99, GPTQ_RESERVE=0.5). Both over budget (~16.35 MB) due to looser clip sigmas. P1b ablates NUM_LOOPS=3 (worse).**

### Ashray's Runs — 2×H100 (rank 4 base, PR #1769)

Teammate experiments on a different base stack (rank 4 + MIN_LR=0.10, vanilla SP8192). Tests normalization techniques.

| Run | Technique | Params | val_bpb (TTT) | Steps | Size | Δ vs baseline |
|-----|-----------|--------|---------------|-------|------|---------------|
| **v1** | **Rank 4 baseline (PR #1769 + MIN_LR=0.10)** | **35.99M** | **1.1374** | **1,379** | **15.99 MB** | **—** |
| v3 | HybridNorm (V-norm + Post-Norm FFN) | 35.99M | — (pre-TTT 1.1564) | 1,334 | — | +0.0108 (worse, TTT killed) |
| v2 | Peri-LN | 35.99M | — (pre-TTT 1.1842) | 1,322 | 15.98 MB | +0.0386 (worse, TTT killed) |

**Key findings:**
- **Peri-LN: BIG regression** (+0.039 BPB). Confirms our independent finding on V2 stack (Paper #22 → NaN on rank 1). Peri-LN fails at this scale regardless of base stack.
- **HybridNorm: also regressed** (+0.011 BPB). V-norm + Post-Norm FFN hurt on rank 4. Rank 4's stack is already heavily normalized (Q/K-norm, ln_scale_factor, resid_mix, attn_scale/mlp_scale) — more normalization conflicts.
- **Normalization axis appears closed** for both rank 1 and rank 4 stacks. Adding norms to already-normalized architectures hurts.
- Rank 4 baseline (1.1374 TTT on 2×H100) is slightly worse than our V2 C6 (1.1622 TTT on 2×H100) — but rank 4 uses phased TTT which we don't have, so pre-TTT comparison is more meaningful: rank 4 pre-TTT 1.1456 vs our V2 C6 pre-TTT ~1.16.

**▲ Ashray runs: 2×H100, seed 42, vanilla SP8192 (kevclark/parameter-golf), 80 train shards, 2026-04-30. Base: train_v1.py (PR #1769 unmodified). v2/v3 TTT killed early due to observed regression.**

**▲ Runs B2-B3: Paper #15 (Small Batch Size), SP8192, 2×H100, FA3, int6+brotli, 2026-04-30. GRAD_ACCUM_STEPS=1 + TRAIN_BATCH_TOKENS=196608 (4× smaller effective batch, 4× more optimizer updates). Best new technique: −0.015 BPB vs C6 baseline. 3,349 steps vs ~1,030 for C6. Beta2 scaling (0.95→0.99) makes no difference. Peri-LN (Paper #22) also tested — went to NaN immediately, output norms destabilize the rank 1 stack.**

**◆ Runs R1-R4: Session 16 EMA deeper sweep + PreQuantTTT, SP8192, 2×H100, FA3, int6+brotli, 2026-04-30. EMA=0.990 new best (R3, 1.1505). PreQuantTTT (R4, 1.0507 TTT) is single biggest gain.**

**◆ Runs N1-N2: Session 17 DiffAttn A/B test, SP8192, 2×H100, FA3, int6+brotli, 2026-04-30. N1 (C6+EMA=0.990+SmallBatch) = 1.1368 (4,221 steps) — new best legal 2×H100 result. EMA+SmallBatch stack: −0.0254 vs C6. N2 (N1+DiffAttn, Paper #19) = 1.1506 (3,292 steps) — FAILS, +0.0138 regression due to 22% fewer steps from 2× FA3 calls. Throughput penalty outweighs attention quality at 36M scale.**

**▼ Run S1: Session 18 Cross-Seq Attn, SP8192, 2×H100, FA3, int6+brotli, 2026-04-30. Same training as N1 (cross-seq is eval-only). Sliding window BPB = 1.1380 (matches N1). †Cross-seq eval hung — sequential window processing (batch_seqs=1) too slow (~3+ hours). TTT BPB not reached. Needs batched implementation.**

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

### Session 12 — V2 Factorial: Rank 1 Fork + Our Techniques (2×H100, 2026-04-28)

Forked rank 1's train_gpt.py (bigbag, 1.0810 BPB) as `train_gpt_v2.py`. Full leaderboard stack: FA3, 11L×512d×8H/4KV, 4×MLP, LeakyReLU², depth recurrence (layers 3-4-5 looped 2×, 17 virtual from 11 unique), parallel residuals (layers 7+), sigmoid skip gates, partial RoPE (16/64 dims), XSA on all 11 layers, MuonEq-R, EMA (0.9965), GPTQ int6+brotli, score-first TTT.

Added our two novelty techniques: gated attention (`GATED_ATTN`) and ResFormer value residual (`VALUE_RESIDUAL_ALPHA`).

3×3 factorial: (PR only vs RF only vs PR+RF) × (No Gate vs Headwise vs Elementwise).

| Run | Config | Gate | RF α | PR Start | Params | Pre-Q BPB | Quant BPB | SW BPB | TTT BPB | Weights | Total | Budget? |
|-----|--------|------|------|----------|--------|-----------|-----------|--------|---------|---------|-------|---------|
| F7 | **PR+RF + No Gate** | none | 0.5 | 7 | 35.94M | 1.1900 | 1.1956 | 1.1790 | **1.1636** | 15.99 MB | 16.04 MB | Tight† |
| F2 | PR + Headwise | head | 0.0 | 7 | 35.99M | 1.1921 | 1.1976 | 1.1812 | 1.1636 | 16.01 MB | 16.06 MB | Tight† |
| F1 | PR + No Gate (CTRL) | none | 0.0 | 7 | 35.94M | 1.1898 | 1.1951 | 1.1790 | 1.1641 | 15.99 MB | 16.04 MB | Tight† |
| F8 | PR+RF + Headwise | head | 0.5 | 7 | 35.99M | 1.1959 | 1.2014 | 1.1850 | 1.1650 | 16.01 MB | 16.06 MB | Tight† |
| F5 | RF + Headwise | head | 0.5 | 999 | 35.99M | 1.1971 | 1.2024 | 1.1861 | 1.1661 | 16.01 MB | 16.06 MB | Tight† |
| F3 | PR + Elementwise | elem | 0.0 | 7 | 38.83M | 1.1951 | 1.2003 | 1.1840 | 1.1665 | 17.21 MB | 17.26 MB | **No** |
| F4 | RF + No Gate | none | 0.5 | 999 | 35.94M | 1.1940 | 1.1996 | 1.1832 | 1.1666 | 15.99 MB | 16.04 MB | Tight† |
| F9 | PR+RF + Elementwise | elem | 0.5 | 7 | 38.83M | 1.1998 | 1.2051 | 1.1888 | 1.1686 | 17.22 MB | 17.27 MB | **No** |
| F6 | RF + Elementwise | elem | 0.5 | 999 | 38.83M | 1.2015 | 1.2069 | 1.1907 | 1.1700 | 17.22 MB | 17.27 MB | **No** |

† "Total" includes 50 KB for our decompressed train_gpt_v2.py. Rank 1's LZMA code is ~16.6 KB. With LZMA code, all non-elementwise runs fit under budget. Weights-only size is what matters for submission.

**3×3 Factor Matrix** (TTT BPB, best in bold):

| | No Gate | Headwise | Elementwise |
|--|---------|----------|-------------|
| **PR only** | 1.1641 | 1.1636 | 1.1665 (over budget) |
| **RF only** | 1.1666 | 1.1661 | 1.1700 (over budget) |
| **PR + RF** | **1.1636** | 1.1650 | 1.1686 (over budget) |

Note: F7 (1.16355) and F2 (1.16361) differ by only 0.00006 BPB — effectively tied within run-to-run noise.

**Key findings:**

1. **Two configs tied for best: F7 (PR+RF, no gate) and F2 (PR, headwise gate)** — 1.16355 vs 1.16361, delta 0.00006 BPB (noise). F7 has the edge on budget (15.99 MB weights, no extra params). F2 adds +21 KB but is more novel for our paper.
2. **ResFormer helps when stacked with PR** — F7 (PR+RF, 1.1636) beats F1 (PR only, 1.1641) by -0.0005. But RF alone is worse: F4 (RF only, 1.1666) loses to F1 by +0.0025. ResFormer is only beneficial as a complement to parallel residuals, not a replacement.
3. **Headwise gate helps on PR, but not on PR+RF** — F2 beats F1 by -0.0005 (headwise helps PR). But F8 (1.1650) is worse than F7 (1.1636) — headwise hurts when RF is also present. The two techniques compete for the same "residual quality" niche.
4. **Elementwise busts budget** — +2.9M extra params from gate projection pushes all elementwise runs to 17.2+ MB. Also slower: 1006-1011 steps vs 1058 for F1. Dead on arrival for 16 MB submission.
5. **RF alone is strictly worse than PR alone** — F4 (RF only, 1.1666) vs F1 (PR only, 1.1641). Consistent across all gate types. Rank 1's parallel residuals + sigmoid skip gates are a stronger residual mechanism.
6. **Budget is extremely tight** — All non-elementwise runs fit under 16 MB (weights only). The 50 KB code overhead from our decompressed file is the main risk; LZMA compression would bring it to ~16.6 KB like rank 1.

**For 8×H100 submission:** C6 (headwise + emb7+eclip15) is the chosen config. 3-seed mean **1.0805 BPP** on 8×H100, all under budget. However, 8×H100 ablation showed headwise gate doesn't improve BPB at this scale (A1 control = 1.0806 ≈ C6 = 1.0818). The rank 1 stack alone is near-optimal; our additions provide paper novelty but not measurable BPB gain.

**Previous 2×H100 assessment (superseded):** F7 (PR+RF) and F2 (headwise) were tied at ~1.1636. Both improved to ~1.08 on 8×H100 but the gap between them and the control vanished.

### Session 13 — V2 Compression Tuning (2×H100, 2026-04-29)

Tested 8 compression knob variants (C1-C8) on the F2 (headwise gate) base to fit under 16 MB. F2 was 16,007,049 bytes (+7 KB over budget). Best result: **C6 (emb7+eclip15) = 1.1622 BPB at 15.71 MB** — under budget with 0.29 MB headroom. See C1-C8 in 2×H100 table above.

### Session 14 — C6 Submission + Ablation (8×H100, 2026-04-29)

Ran C6 config (headwise + emb7+eclip15) on 8×H100 for PG submission, plus ablation runs.

**Part A: 3-seed C6 submission** — seeds 42, 1337, 2025. Mean **1.0805 BPB** (std ±0.0012). All under 16 MB, train <600s, eval <600s. See 3-seed table above.

**Part B: Ablation** (seed 42, all 3 runs completed):
- A1 (F1 control, no additions): **1.0806 BPB**, 15,977,755 bytes
- A2 (F7 PR+RF, α=0.5): **1.0828 BPB**, 15,983,964 bytes
- A3 (F2 headwise, default compression): **1.0801 BPB**, 15,993,169 bytes (total 16,043,196 — over budget)

**Key findings:**
1. **Headwise gate helps** — A3 (1.0801) beats A1 (1.0806) by -0.0005 BPB. Consistent with 2×H100 (F2 vs F1 = -0.0005).
2. **Compression tuning costs BPB** — C6 (1.0818) is +0.0017 worse than A3 (1.0801). The emb7+eclip15 settings trade quality for size.
3. **ResFormer hurts** — A2 (1.0828) is worst of all four. α=0.5 doesn't help at this scale.
4. **A3 is over total budget** — weights fit (15.99 MB) but total with code is 16.04 MB. Same problem as F2 on 2×H100.

**Scaling:** 2×H100 → 8×H100 dramatically improved all configs. F1: 1.1641 → 1.0806 (−0.0835). F2: 1.1636 → 1.0801 (−0.0835). C6: 1.1622 → 1.0818 (−0.0804). Technique deltas preserved.

**Decision:** Hold submission. We match SOTA (1.0805 vs 1.0810) but don't clear the ≥0.005 nats threshold. Keep technique secret until we can widen the gap.

### Session 15 — C6 Fine-Tuning Sweep (2×H100, 2026-04-29)

19-run hyperparameter sweep on C6 base (headwise + emb7+eclip15). All 19 runs completed.

| Run | Group | Setting | Rank 1 Default | TTT BPB | SW BPB | Pre-Q BPB | Weights | vs C6 (1.1622) |
|-----|-------|---------|----------------|---------|--------|-----------|---------|----------------|
| **E1** | **EMA** | **decay=0.995** | **0.9965** | **1.1562** | **1.1607** | **1.1696** | **15,706,586** | **-0.0060** |
| W2 | WD | wd=0.10 | 0.095 | 1.1619 | 1.1777 | 1.1877 | 15,706,946 | -0.0003 |
| T8 | TTT | e7 lr0.005 | e3 lr0.005 | 1.1624 | 1.1803 | 1.1901 | 15,707,547 | +0.0002 |
| W3 | WD | wd=0.11 | 0.095 | 1.1628 | 1.1804 | 1.1905 | 15,707,031 | +0.0006 |
| T9 | TTT | e7 lr0.01 | e3 lr0.005 | 1.1629 | 1.1815 | 1.1912 | 15,706,433 | +0.0007 |
| T7 | TTT | e7 lr0.003 | e3 lr0.005 | 1.1631 | 1.1800 | 1.1898 | 15,706,542 | +0.0009 |
| T3 | TTT | e3 lr0.01 | e3 lr0.005 | 1.1633 | 1.1808 | 1.1906 | 15,707,885 | +0.0011 |
| Q2 | QK-Gain | qkg=5.75 | 5.25 | 1.1634 | 1.1797 | 1.1895 | 15,706,309 | +0.0012 |
| T5 | TTT | e5 lr0.005 | e3 lr0.005 | 1.1634 | 1.1810 | 1.1907 | 15,707,393 | +0.0012 |
| Q3 | QK-Gain | qkg=6.0 | 5.25 | 1.1636 | 1.1802 | 1.1900 | 15,706,643 | +0.0014 |
| T6 | TTT | e5 lr0.01 | e3 lr0.005 | 1.1636 | 1.1826 | 1.1924 | 15,707,087 | +0.0014 |
| Q1 | QK-Gain | qkg=5.5 | 5.25 | 1.1642 | 1.1806 | 1.1905 | 15,706,206 | +0.0020 |
| T4 | TTT | e5 lr0.003 | e3 lr0.005 | 1.1643 | 1.1806 | 1.1904 | 15,707,705 | +0.0021 |
| D1 | Warmdown | frac=0.80 | 0.72 | 1.1645 | 1.1804 | 1.1899 | 15,707,737 | +0.0023 |
| T2 | TTT | e3 lr0.005 | e3 lr0.005 | 1.1650 | 1.1818 | 1.1916 | 15,706,681 | +0.0028 |
| W1 | WD | wd=0.08 | 0.095 | 1.1650 | 1.1796 | 1.1892 | 15,708,632 | +0.0028 |
| T1 | TTT | e3 lr0.003 | e3 lr0.005 | 1.1665 | 1.1826 | 1.1923 | 15,707,582 | +0.0043 |
| E2 | EMA | decay=0.997 | 0.9965 | 1.1690 | 1.1967 | 1.2069 | 15,706,990 | +0.0068 |
| E3 | EMA | decay=0.999 | 0.9965 | 1.3475 | 2.9634 | 2.9498 | 15,718,492 | +0.1853 |

**Key findings:**
1. **EMA=0.995 is the big winner** — 1.1562 BPB, -0.0060 below C6 baseline. More aggressive averaging (lower decay) helps at this training duration. If the delta holds at 8×H100, projects to ~1.0745 BPB — **would clear SOTA threshold** (need ~1.0760).
2. **Weight Decay=0.10 marginal** — 1.1619, -0.0003 vs C6. Tiny but could stack with EMA=0.995.
3. **TTT tuning doesn't help** — T8 (e7, lr0.005) is best at 1.1624 (+0.0002), essentially tied with C6. More epochs help slightly (7 > 5 > 3) but gains are tiny and not worth the extra eval time. Default (3 epochs, lr=0.005) is near-optimal.
4. **QK-Gain, Warmdown changes all hurt** — rank 1's defaults (5.25, 0.72) are already optimal.
5. **EMA sensitivity is extreme** — 0.995 (best) → 0.997 (worse) → 0.999 (catastrophic 1.3475). Sweet spot is tighter averaging.

**TODO:** Run EMA=0.995 + WD=0.10 combo on 8×H100. If -0.006 delta holds, we clear SOTA.

### Session 16 — EMA Deeper Sweep + PreQuantTTT (2×H100, 2026-04-30)

#### Phase 1: EMA Deeper Sweep

| Run | Setting | TTT BPB | SW BPB | Pre-Q BPB | Weights | vs C6 (1.1622) | vs E1 (1.1562) |
|-----|---------|---------|--------|-----------|---------|----------------|----------------|
| R1 | EMA=0.995 + WD=0.10 | 1.1559 | 1.1606 | 1.1696 | 15,706,897 | -0.0063 | -0.0003 |
| R2 | EMA=0.993 | 1.1526 | 1.1546 | 1.1626 | 15,707,959 | -0.0096 | -0.0036 |
| **R3** | **EMA=0.990** | **1.1505** | **1.1521** | **1.1591** | **15,708,141** | **-0.0117** | **-0.0057** |

**Key finding:** EMA keeps improving as decay decreases. 0.990 is new best, nearly 2× the gain of 0.995 (-0.0117 vs -0.0060). Projected 8×H100: ~1.069 BPB.

#### Phase 2-3: PreQuantTTT + Compression

| Run | Setting | TTT BPB | SW BPB | Pre-Q BPB | PostPQ BPB | Weights | Budget |
|-----|---------|---------|--------|-----------|------------|---------|--------|
| **R4** | **PreQuantTTT (brotli)** | **1.0507** | **1.0765** | **1.1591** | **1.0156** | **15,705,262** | **Yes** |
| R5 | PreQuantTTT (pergroup) | crashed | crashed | 1.1595 | 1.0150 | 15,724,064 | Yes |

**Key findings:**
1. **PreQuantTTT is transformative** — takes pre-Q 1.1591 → post-PQ 1.0156 BPB (-0.1435). On 2×H100, post-quant TTT gives **1.0507** — better than our 8×H100 C6 result (1.0805).
2. R5 crashed at `deserialize()` due to `torch.load(..., weights_only=True)` default in PyTorch 2.11. Fix: add `weights_only=False`.
3. **Projected 8×H100:** ~0.97-1.00 BPB — would beat current SOTA (1.0136).

### Session 18 — SOTA Hparam Adoption Phase 1 (8×H100, 2026-04-30)

Adopted 6 hyperparameter overrides from SOTA PR #1855 (codemath3000, 1.0611 BPB) into our C6 stack. Zero code changes — env vars only. Also ablated depth recurrence (NUM_LOOPS=3 vs default 2).

**Hparam overrides from PR #1855:**

| Parameter | C6 Value | P1 Value | Source |
|-----------|----------|----------|--------|
| WARMDOWN_FRAC | 0.72 | 0.85 | SOTA #1855 |
| MIN_LR | 0.0 | 0.10 | SOTA #1855 |
| MATRIX_CLIP_SIGMAS | 12.85 | 11.5 | SOTA #1855 (looser MLP clipping) |
| EMBED_CLIP_SIGMAS | 15.0 | 14.0 | SOTA #1855 (tighter embed) |
| BETA2 | 0.95 | 0.99 | SOTA #1855 (AdamW embed/scalar) |
| GPTQ_RESERVE_SECONDS | 12 | 0.5 | SOTA #1855 (more training time) |

**Results (8×H100, seed 42):**

| Run | Config | Params | Steps | Pre-Q post-EMA BPB | SW BPB | TTT BPB | Size | Budget? |
|-----|--------|--------|-------|---------------------|--------|---------|------|---------|
| **P1a** | **C6 + 6 SOTA hparam overrides** | **35.99M** | **4,587** | **1.0844** | **1.0783** | **1.0769** | **16.35 MB** | **No** |
| P1b | C6 + SOTA hparams + NUM_LOOPS=3 | 35.99M | 4,126 | 1.0874 | 1.0809 | 1.0793 | 16.36 MB | **No** |
| *C6* | *Reference (3-seed mean)* | *35.99M* | *4,467* | — | — | *1.0805* | *15.70 MB* | *Yes* |

**Key findings:**

1. **SOTA hparams work: −0.0036 BPB vs C6** (P1a 1.0769 vs C6 1.0805). All 6 overrides combined give a meaningful improvement with zero code changes.
2. **Over budget: 16.35 MB** (P1a) and 16.36 MB (P1b) — both exceed 16,000,000 byte limit. MATRIX_CLIP_SIGMAS=11.5 (looser than C6's 12.85) allows larger weight values that compress worse under brotli. Need to tighten clip sigmas to fit.
3. **NUM_LOOPS=3 hurts** — P1b (1.0793) is +0.0024 worse than P1a (1.0769). Extra depth recurrence pass (4 passes vs 3) reduces throughput: 4,126 steps vs 4,587 (−10%). The throughput penalty outweighs the depth benefit. Keep NUM_LOOPS=2.
4. **P1a eval time: 349s** (within 600s limit). P1b eval time: 455s — NUM_LOOPS=3 also slows eval by 30%.
5. **Next step:** Tighten MATRIX_CLIP_SIGMAS (try 12.0 or 12.5) to fit under 16 MB while retaining most of the BPB gain.

### Session 19 — P3 SOTA Runs (8×H100, 2026-04-30)

Forked PR #1851 (@aquariouseworkman, 1.0611 BPB) as base. Applied 4 novel contributions: headwise gated attention, EMA=0.990, small batch (ga=1, 196K tokens), EMBED_BITS=6. CaseOps ON (via symlinked data), SmearGate OFF, LQER ON, QK-Gain 5.0.

**Config (key overrides from PR #1851 defaults):**

| Parameter | PR #1851 Default | P3 Value | Source |
|-----------|-----------------|----------|--------|
| GATED_ATTN_ENABLED | 0 | 1 | James Vo (novel) |
| EMA_DECAY | 0.9965 | 0.990 | James Vo (novel finding) |
| GRAD_ACCUM_STEPS | default | 1 | James Vo (Paper #15) |
| TRAIN_BATCH_TOKENS | 786432 | 196608 | James Vo (Paper #15) |
| EMBED_BITS | 7 | 6 | James Vo (novel) |
| CASEOPS_ENABLED | 1 | 0 (but CaseOps data used via symlinks) | Active via data path |

**3-Seed Results (8×H100, seeds 42/1337/2025):**

| Seed | Pre-Q BPB | Quant BPB | TTT BPB | Size (bytes) | Budget? |
|------|-----------|-----------|---------|--------------|---------|
| 42   | 1.0025    | 1.0205    | 1.0069  | 15,975,827   | Yes |
| 1337 | 1.0017    | 1.0190    | 1.0057  | 15,973,108   | Yes |
| 2025 | 1.0030    | 1.0206    | 1.0073  | 15,973,714   | Yes |
| **Mean** | **1.0024** | **1.0200** | **1.0066** | **~15,974K** | **Yes** |
| **Std** | **0.0007** | **0.0009** | **±0.0009** | — | — |

12,382 steps in 596s (under 600s). TTT eval: 353-389s (under 600s). All artifacts under 16 MB.

**Key findings:**

1. **~~P3 IS THE NEW SOTA~~ — RETRACTED.** Original 1.0066 BPB was an artifact of inflated byte denominator. Corrected BPB (with `CASEOPS_ENABLED=1` sidecar): **1.0972** (seed 42) — **worse than C6 (1.0805)**. The val_loss (~2.401) is genuine but BPB was computed against ~164.6M CaseOps-transformed bytes instead of ~151M canonical raw bytes.
2. **EMA=0.990 + small batch do NOT help on PR #1851 stack at 8×H100** — corrected P3 (1.0972) is worse than C6 (1.0805) by +0.0167. Consistent with L1/L2 results on @bigbag stack. The apparent improvement was entirely the byte counting artifact.
3. **EMBED_BITS=6 trades ~0.013 quant gap for ~1 MB savings** — enables headwise gate to fit under 16 MB (~15.97 MB).
4. **CaseOps byte accounting is critical** — `CASEOPS_ENABLED=0` with CaseOps tokenizer silently inflates the byte denominator by ~9%. Must use `CASEOPS_ENABLED=1` (loads sidecar) for correct BPB when using CaseOps data.

**Compliance (verified):** train_under_600s (596s), artifact_under_16mb (all <15,976,000), eval_under_600s (353-389s), no_pre_quant_ttt, score_first_ttt, three_seeds (42/1337/2025).

Submitted as PR #2071 to openai/parameter-golf.

### PR #2071 Legality Feedback — Byte Accounting Concern

A reviewer raised a concern about our P3 submission. The issue is **not** probability normalization (C2) or score-after-update (C3) — both are clean. The concern is that the **byte denominator used in BPB calculation may be inflated** due to our hybrid CaseOps setup.

#### The Problem: Two Byte Counting Paths

The evaluation code (`train_gpt.py`) has two byte counting methods:

1. **Sidecar path** (when `caseops_enabled=True`): Loads a pre-computed per-token byte sidecar file (`fineweb_val_bytes_*.bin`) that stores the **canonical raw-text byte budget** for each token. This is the correct denominator — it counts how many raw FineWeb bytes each CaseOps token represents.

2. **LUT path** (when `caseops_enabled=False`): Calls `build_sentencepiece_luts()` which computes byte counts from the tokenizer's vocabulary pieces via `len(piece.encode("utf-8"))`. For a regular SP8192 tokenizer, this gives the correct raw byte count. For a **CaseOps tokenizer**, this gives the byte count of the CaseOps-**transformed** text (which includes case markers), not the raw text.

Our P3 runs used `CASEOPS_ENABLED=0` but loaded the CaseOps tokenizer and CaseOps-tokenized data via symlinks. This means the code took the **LUT path** with a **CaseOps tokenizer**, computing byte counts from CaseOps-transformed pieces rather than raw FineWeb bytes.

#### The Math

From `train_seed42.log`:
```
caseops_enabled: False
val_tokens: 47,851,520
quantized_ttt_phased val_loss:2.40073153 val_bpb:1.00692894
```

The BPB formula is: `val_bpb = (val_loss / ln(2)) × (val_tokens / val_bytes)`

Solving for the byte denominator used:
```
val_bytes = val_loss / ln(2) × val_tokens / val_bpb
          = 2.40073153 / 0.693147 × 47,851,520 / 1.00692894
          ≈ 164,594,398
```

The reviewer states the canonical raw FineWeb validation byte count is **153,880,891**. If BPB is recomputed against canonical bytes:
```
corrected_bpb = 2.40073153 / ln(2) × 47,851,520 / 153,880,891
              ≈ 1.0770 BPB
```

This means the **~7% byte inflation** from CaseOps token pieces (`164.6M / 153.9M ≈ 1.070×`) would shift our reported 1.0066 to approximately **1.077 BPB** — close to our C6 result (1.0805) rather than a significant improvement.

#### Why This Affects Only Our Setup

| Setup | Byte Counting | Denominator | Correct? |
|-------|--------------|-------------|----------|
| Regular SP8192 + `CASEOPS_ENABLED=0` | LUT from regular tokenizer | Raw bytes | Yes |
| CaseOps + `CASEOPS_ENABLED=1` (PR #1851 default) | Sidecar file | Canonical raw bytes | Yes |
| **CaseOps data + `CASEOPS_ENABLED=0` (our P3)** | **LUT from CaseOps tokenizer** | **CaseOps-transformed bytes** | **No — inflated** |

The CaseOps tokenizer's vocabulary pieces encode CaseOps-transformed text (lowercase + case markers). When `build_sentencepiece_luts()` counts `len(piece.encode("utf-8"))` for these pieces, it includes the bytes from case markers that don't exist in the raw text. The sidecar exists precisely to provide the correct raw byte count for each CaseOps token, but our `CASEOPS_ENABLED=0` flag bypassed it.

PR #1851's own runs use `CASEOPS_ENABLED=1` and load the sidecar, so they have the correct byte denominator. Other non-CaseOps submissions use the regular tokenizer's LUT, which also gives correct raw byte counts. Our hybrid setup is the only configuration that produces an inflated denominator.

#### What's NOT Affected

The reviewer confirmed:
- **No C2 violation** — eval uses standard `F.cross_entropy()` over full vocab; fused CE uses normal log-sum-exp. Probability distributions are properly normalized.
- **No C3 violation** — phased TTT scores tokens under `torch.no_grad()` BEFORE LoRA gradient updates. Score-first ordering is correct.
- **The model quality is real** — the cross-entropy loss (`val_loss = 2.4007`) is genuinely good. The issue is purely how that loss is converted to BPP via the byte denominator.

#### Rerun Results (2026-05-01) — CONFIRMED

Reran P3 with `CASEOPS_ENABLED=1` + explicit `DATA_PATH` + byte sidecar file downloaded from `romeerp/parameter-golf-caseops-v1`. No symlinks. Sidecar file provides canonical raw byte counts per token.

**Seed 42 rerun:**
```
caseops_enabled: True
diagnostic pre-quantization post-ema val_loss:2.39422591 val_bpb:1.09399677
quantized_ttt_phased val_loss:2.40115802 val_bpb:1.09723551 eval_time:459203ms
```

| Metric | Original P3 (inflated) | Corrected P3 (sidecar) | Delta |
|--------|----------------------|----------------------|-------|
| val_loss (TTT) | 2.4007 | 2.4012 | ~same |
| val_bpb (TTT) | **1.0069** | **1.0972** | **+0.0903** |
| Byte denominator | ~164.6M (LUT) | ~151M (sidecar) | -8.3% |

The val_loss is essentially identical — confirming the model quality didn't change. The entire BPB difference is from the byte denominator correction.

**Corrected P3 (1.0972) is worse than C6 (1.0805) by +0.0167.** The reviewer was correct — the original 1.0066 was an artifact of the inflated byte count. The reviewer's estimate (~1.077) was directionally correct but slightly off because their assumed canonical byte count (153.9M) differs from the actual sidecar total (~151M).

---

**Our best legal run: C6 at 1.0805 BPB** (3-seed mean). P3 retracted due to byte accounting error. External SOTA: 1.0611 (codemath3000, PR #1855).

### Official Leaderboard (as of 2026-04-30)

| Rank | BPB | Author | Key Techniques |
|-----:|------:|--------|---------------|
| ~~NEW~~ | ~~1.0066~~ | ~~Us (P3, PR #2071)~~ | ~~RETRACTED — inflated byte denominator. Corrected: 1.0972~~ |
| 1 | 1.0585 | ndokutovich (#1967) | N-gram Tilt + LeakyReLU 0.3 |
| 2 | 1.0586 | andrewbaggio1 (#1953) | Long-context 2560 + no_qv TTT mask + QK_GAIN 5.25 |
| 3 | 1.0593 | alertcat (#1945) | AWQ-lite + Asymmetric Logit Rescale |
| 4 | 1.0600 | Christopher-Lee-McClendon (#1950) | #1934 reproduction (GPTQ_RESERVE=5.5) |
| 5 | 1.0604 | AayushBaniya2006 (#1956) | #1908 reproduction |
| 6 | 1.0609 | aquariouseworkman (#1946) | AWQ-lite mixed-precision GPTQ |
| prev | 1.0611 | codemath3000 (#1855) | BOS-Fixed SmearGate + LQER + SparseAttnGate + per-group lrzip + 9 greedy HP overrides |
| 7 | 1.0624 | TimS-ml (#1948) | Leaky ReLU Slope + GPTQ Reverse-Cholesky |
| 8 | 1.0687 | MarioPaerle (#1941) | Per-block MLP output gate |
| — | — | — | — |
| old 1 | 1.0810 | bigbag | SP8192, 3-layer recurrence, parallel residuals, QK-Gain 5.25, legal TTT |
| | 1.0805 | Us (C6, 3-seed mean) | V2: rank 1 fork + headwise gated attn + emb7+eclip15 |
| old 2 | 1.0822 | aryanbhosale | SP8192, parallel residuals, score-first TTT |
| old 3 | 1.0828 | dexhunter | SP8192, QK-Gain 5.0, legal score-first TTT |

### Where We Stand

- **P4b is NEW BEST: 1.0621 BPB** (CaseOps + SOTA hparams + headwise gate, 15.98 MB, under budget)
- **Gap to external SOTA (1.0611, PR #1855):** +0.001 BPB (essentially tied)
- **Delta vs C6:** −0.0184 BPB (P4b beats C6 decisively)
- **Gap to baseline (1.2244):** −0.1623 BPB
- P3 RETRACTED — original 1.0066 was inflated byte denominator. Corrected: 1.0972 (worse than C6)
- **Lesson learned:** CaseOps byte accounting requires `CASEOPS_ENABLED=1` with sidecar file for correct BPB. Symlink + `CASEOPS_ENABLED=0` silently inflates denominator by ~9%.

### Submission Strategy

**We are submitting P3 as a RECORD submission.**

- Our P3 mean: **1.0066 BPB** (std ±0.0009)
- Previous SOTA: 1.0136 BPB (PR #1958 — C3 violation, withdrawn)
- Verified legal SOTA: 1.0611 BPB (PR #1855 codemath3000)
- We beat legal SOTA by **0.0545 nats** (well above ≥0.005 threshold)
- New PR to openai/parameter-golf with 3-seed logs

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
| **Small Batch (Paper #15)** | **-0.015 BPB** (1.1419 vs 1.1572) | ga=1 + TRAIN_BATCH_TOKENS÷4 gives 4× more optimizer updates in same wall clock. 3,349 steps vs ~1,030. Beta2 scaling irrelevant. Paper: "Small Batch Size Training" (NeurIPS 2024). |
| **EMA=0.990** | **-0.0117 BPB** vs C6 (1.1505 vs 1.1622) | More aggressive weight averaging helps at short training durations. Nearly 2× the gain of 0.995. Sweet spot shifts lower with fewer training steps. |
| **PreQuantTTT** | **-0.1435 BPB** (1.1591→1.0156 post-PQ) | 21 epochs AdamW on val before GPTQ. Freezes blocks 0-1 + embeddings, cosine LR 5e-4→5e-5. Single biggest technique gain in entire project. Source: okezue (PG PR #1958). |

## Techniques That Didn't Work

| Technique | Expected impact | Actual result | Why it failed |
|---|---|---|---|
| Embedding compression (emb7+eclip15) | Fit headwise gate under 16 MB | +0.0017 BPB cost (A3=1.0801 vs S1=1.0818 on 8×H100) | int7 embeddings + tighter clipping trades too much quality for the ~0.29 MB size savings. Need a better compression approach to fit headwise gate under budget. |
| ResFormer (α=0.5, 8×H100) | Blend V₀ into all layers | +0.0022 worse on 8×H100 (A2=1.0828 vs A1=1.0806) | Helped on 2×H100 V1 GPTQ stack (1.2536 vs 1.2584) but hurts on the rank 1 V2 stack at 8×H100 scale. |
| Gated Attention (elementwise) | Better BPB than headwise | Best BPB (1.2338 at dim=448) but 16.67 MB over budget; dim=416 fits but 1.2447 worse than Run A | Elementwise needs dim≥448 to beat headwise, but that's over 16 MB. Shrinking dim kills the gain. No sweet spot found. |
| MQA on SP8192 | Faster inference, smaller model | 1.2509 BPB at dim=448 — 0.0098 worse than GQA (Run A) | Confirmed on SP8192 (Session 8) after SP1024 (Run 4). Fewer KV heads = worse quality at 17M scale. |
| QK-Gain 5.0 (on SP1024) | Better attention scaling | 1.2719 BPB (worse than headwise 1.2653) | 15% slower steps (210ms vs 182ms), higher VRAM (13GB vs 10GB). QK-Gain 5.0 likely needs SP8192 to be effective. |
| SLM / Rho-1 (all ratios) | Better per-step learning by filtering easy tokens | k=0.6: +0.155 BPB, k=0.8: +0.024, k=0.95: +0.002 — ALL worse than no-SLM | At 17M params, model needs every gradient signal. Rho-1 paper tested at 1B+; doesn't transfer down. Simple loss-threshold (Option A) too crude without reference model. Paper: "Not All Tokens Are What You Need" (NeurIPS 2024). |
| LR Warmup (Paper #16) | Better training stability | 2%: +0.0024, 5%: +0.0042, 10%: +0.0066 — ALL worse, monotonically | Rank 1 correctly skips warmup. With MuonEq-R optimizer and existing momentum warmup, LR warmup adds redundant ramp. Paper: "On the Role of LR Warmup" (ICML 2025). |
| Structured FFN (Paper #5) | Reduce MLP params via low-rank + block-diagonal | r=0.5/b=4: +0.043 BPB, r=0.75/b=8: +0.050 BPB — saves 30-56% params but quality collapses | Paper tested at 125M+; structure constraints too lossy at 36M scale where every param counts. Paper: "Structured FFN" (NeurIPS 2024). |
| Peri-LN (Paper #22) | Better norm placement | Immediate NaN — training collapses | Output norms on attn+MLP conflict with existing attn_scale/mlp_scale + depth-dependent ln_scale_factor. Destabilizes the rank 1 stack. Paper: "Peri-LN" (ICML 2025). |

## Key Insights

_High-level takeaways that apply beyond the competition._

1. 2 GPUs got to val_bpb 1.30 in 10 min — 8 GPUs should process ~4× more tokens in the same window, likely pushing below 1.22
2. Model still improving at wall clock cutoff — not converged, more throughput = better score
3. int8+zlib compression is essentially free (1.3033 → 1.3045, only +0.001 BPB degradation)
4. SP8192 dataset is NOT in the official PG repo (`willdepueoai/parameter-golf`). It's hosted on Kevin Clark's fork: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80`. All top 5 submissions (ranks 1-5) use this source.
5. **SLM (Rho-1) doesn't work at 17M scale** — validated in Session 7 with working code. Every ratio (k=0.6 to k=0.95) hurts. Small models need all tokens; the paper's 1B+ results don't transfer down.
6. **Techniques stack cleanly** — SP8192 + TTT + LeakyReLU² + headwise + QKG5 all combine without interference. Best 2×H100: 1.2411 BPB (Run A).
7. **3-seed reproducibility confirmed** — V2 C6 on 8×H100: mean 1.0805 BPP (std ±0.0012). V1 combo slim on 8×H100: mean 1.2073 (std ±0.0006). Results are stable across random seeds.
8. **Total cost: ~$1,165 across 130+ experiments** — systematic ablation approach validated each technique individually before stacking.
9. **Elementwise gated attention: best BPB but no budget-legal sweet spot** — E1 (dim=448, 1.2338) is the best 2×H100 BPB ever but 0.67 MB over. dim=416 fits but loses all quality gain. MQA also confirmed worse on SP8192.
10. **P3 IS the new SOTA** — 1.0066 BPB (3-seed mean), beats legal SOTA (1.0611, PR #1855) by −0.0545 nats. PR #1851 fork + headwise gate + EMA=0.990 + small batch + emb6. Previous C6 (1.0805) now superseded.
11. **Small batch size is the biggest V2 technique win (Paper #15)** — removing gradient accumulation (ga=4→1) and reducing TRAIN_BATCH_TOKENS by 4× gives 3.3× more optimizer updates in the same wall clock. Result: −0.015 BPB on 2×H100 (1.1419 vs 1.1572). Beta2 scaling (0.95→0.99) makes no difference. New best 2×H100 BPB ever.
12. **Peri-LN (Paper #22) kills training** — output norms on attn+MLP cause immediate NaN. Conflicts with existing attn_scale/mlp_scale + depth-dependent ln_scale_factor. Do not use on rank 1 stack.
13. **LR warmup hurts (Paper #16)** — tested 2%, 5%, 10% warmup fractions. All worse, monotonically: +0.0024, +0.0042, +0.0066 BPB. Confirms rank 1's design choice to skip LR warmup.
14. **Structured FFN fails at V2 scale (Paper #5)** — low-rank + block-diagonal FFN saves 30-56% params but +0.04-0.05 BPB degradation. Paper tested at 125M+; doesn't transfer to 36M.
15. **Headwise gate effect preserved at scale, but compression costs wipe it out** — A3 (headwise, 1.0801) beats A1 (control, 1.0806) by -0.0005 BPB on 8×H100, same delta as 2×H100. But C6's compression tuning (emb7+eclip15) adds +0.0017 BPB cost. Net effect of C6 vs A1: +0.0012 worse. ResFormer (α=0.5) hurts at scale (+0.0022). Need better compression to realize the headwise gate gain within budget.
16. **EMA=0.990 is optimal** — deeper sweep (Session 16) confirms more aggressive weight averaging helps at this training duration. Nearly 2× the gain of 0.995 (−0.0117 vs −0.0060 below C6). Sweet spot shifts lower with fewer training steps.
17. **PreQuantTTT is transformative** — 21 epochs AdamW on val before GPTQ gives −0.1435 BPB (1.1591→1.0156 post-PQ). On 2×H100, R4 post-quant TTT (1.0507) beats our 8×H100 C6 (1.0805). Single biggest technique gain in entire project. Projected 8×H100: ~0.97-1.00 BPB.
18. **SOTA hparam overrides give −0.0036 BPB but bust budget** — 6 env-var-only changes from PR #1855 (WARMDOWN=0.85, MIN_LR=0.10, MATRIX_CLIP_SIGMAS=11.5, EMBED_CLIP_SIGMAS=14.0, BETA2=0.99, GPTQ_RESERVE=0.5) improve P1a to 1.0769 on 8×H100. But looser MATRIX_CLIP_SIGMAS inflates compressed size to 16.35 MB. Need to find clip sigma sweet spot between quality and budget.
19. **NUM_LOOPS=3 (4 depth recurrence passes) hurts** — P1b (1.0793) is +0.0024 worse than P1a (1.0769). Extra pass costs 10% throughput (4,126 vs 4,587 steps) and 30% longer eval (455s vs 349s). Keep NUM_LOOPS=2.
20. **P3 RETRACTED — byte accounting error confirmed.** Original 1.0066 BPB was artifact of inflated byte denominator. `CASEOPS_ENABLED=0` + CaseOps tokenizer used LUT byte counting (~164.6M) instead of sidecar (~151M). Rerun with `CASEOPS_ENABLED=1` gives **1.0972 BPB** (seed 42) — worse than C6 (1.0805) by +0.0167. The val_loss (~2.401) was real but the BPB conversion was wrong. Root cause: symlink setup on pod loaded CaseOps tokenizer but bypassed byte sidecar.
21. **EMBED_BITS=6 trades ~0.013 quant gap for ~1 MB savings** — C2 (emb6, 2×H100) was 1.1735 vs C6 (emb7, 1.1622), +0.0113 gap. But the ~1 MB savings enables headwise gate to fit comfortably under 16 MB (~15.97 MB vs ~15.99 MB with emb7).
22. **EMA=0.990 + small batch do NOT transfer to PR #1851 stack at 8×H100** — corrected P3 (1.0972) is worse than C6 (1.0805). The apparent "transfer" in the original P3 was entirely the byte counting artifact. Consistent with L1/L2 results on @bigbag stack: aggressive EMA + small batch hurt at 8×H100 scale regardless of base stack.
23. **CaseOps byte accounting is a silent trap** — `CASEOPS_ENABLED=0` with CaseOps tokenizer (via symlink) inflates the byte denominator by ~9% without any warning. The code produces plausible-looking BPB numbers that are systematically too low. Must use `CASEOPS_ENABLED=1` with sidecar file for correct reporting. This was the single most costly mistake of the project.

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

---

## Log file index

Each entry in this document corresponds to a `.txt` log file in `parameter-golf/logs/`. Filenames are prefixed with the run label used above so `ls parameter-golf/logs/ | grep <label>` finds the source data instantly.

**Naming legend:** `_OVER` over budget · `_FAILED` / `_RETRACTED` invalid result · `_INCOMPLETE` truncated · `_DUP` byte-identical duplicate · `_INFLATED` byte-denominator artifact · `_C3VIOL` Parameter Golf C3 rule violation (PreQuantTTT) · `INVALID_S6_*` SLM code absent on pod · suffix `_alt` distinct rerun · suffix `-rerun` repeat on different hardware.

### 8×H100 official runs

| Run | Log filename |
|-----|-------------|
| C6 (3-seed) | `C6_v2_c6_seed42.txt`, `C6_v2_c6_seed1337.txt`, `C6_v2_c6_seed2025.txt` |
| A1 ablation | `A1_v2_c6_nogptqtune_8gpu.txt`, `A1_v2_f1_8gpu_alt.txt` |
| A2 ablation | `A2_v2_f7_8gpu.txt` |
| L1 EMA=0.990 | `L1_legal_ema090.txt`, `L1_legal_ema090_seed42.txt` |
| L2 EMA=0.990 + small batch | `L2_legal_ema090_smallbatch.txt`, `L2_legal_ema090_smallbatch_seed42.txt` |
| P1a SOTA hparams | `P1a_hparam_sota.txt`, `P1a_hparam_sota_seed42.txt` |
| P1b NUM_LOOPS=3 | `P1b_hparam_sota_loop3.txt`, `P1b_hparam_sota_loop3_seed42.txt` |
| P1c clip12 | `P1c_p4_clip12.txt` |
| P3 (retracted) | `P3_inflated_seed42_RETRACTED.txt`, `P3_inflated_seed1337_RETRACTED.txt`, `P3_inflated_seed2025_RETRACTED.txt` |
| P3-fix corrected | `P3-fix_corrected_seed42.txt`, `P3-fix_corrected_seed1337.txt`, `P3-fix_corrected_seed42_incomplete.txt` |
| **P4b NEW BEST** | `P4b_caseops_seed42.txt` |
| Run 11 (V1, 8×H100) | `Run11_sp8192_combo_slim.txt`, `Run11_sp8192_combo_slim_8gpu_alt.txt` |
| Run 11 3-seed | `Run11-3seed_sp8192_combo_slim_seed42.txt`, `Run11-3seed_sp8192_combo_slim_seed1337.txt`, `Run11-3seed_sp8192_combo_slim_seed2025.txt` |

### V1 numbered runs (Sessions 1–7, 2×H100)

| Run | Log filename |
|-----|-------------|
| Run 1 | `Run01_baseline_sp1024.txt` |
| Run 2 headwise | `Run02_gated_attn_headwise.txt` |
| Run 3 elementwise (over) | `Run03_gated_attn_elementwise_OVER.txt` |
| Run 4 MQA (over) | `Run04_mqa_OVER.txt` |
| Run 5 INVALID | `Run05_INVALID_stale_env.txt` |
| Run 6 GQA baseline | `Run06_gqa_baseline.txt` |
| Run 7 LeakyReLU² | `Run07_leaky_relu2.txt`, `Run07-rerun_leaky_relu2_2gpu.txt` |
| Run 8 LeakyReLU² + headwise | `Run08_leaky_relu2_headwise.txt` |
| Run 9 headwise + QK-Gain 5.0 | `Run09_headwise_qkgain5.txt`, `Run09-rerun_headwise_qkgain5_2gpu.txt` |
| Run 10 SP8192 combo (over) | `Run10_sp8192_combo_OVER.txt` |
| Run A / D / H | `RunA_sp8192_combo_slim_2gpu.txt`, `RunD_sp8192_combo_slim_2gpu.txt`, `RunH_sp8192_combo_slim_nottt.txt` |

### Session 8 — Elementwise + MQA sweep

| Run | Log filename |
|-----|-------------|
| E1 (over) | `S8-E1_elem_dim512_OVER.txt` |
| E2 | `S8-E2_elem_dim448.txt` |
| dim=480 (extra) | `S8-extra_elem_dim480.txt` |

### Session 9 — GPTQ validation

| Run | Log filename |
|-----|-------------|
| G2 best | `S9-G2_gptq_int7_train.txt` |
| G3 | `S9-G3_gptq_int6_ar.txt` |
| G4 | `S9-G4_gptq_int6_train.txt` |

### Session 10 — Benchmark sweep (dim / layers / attn)

| Run | Log filename |
|-----|-------------|
| D1 dim=448 | `S10-D1_bench_dim448.txt` |
| D2 dim=512 | `S10-D2_bench_dim512.txt` |
| D3 dim=768 (over) | `S10-D3_bench_dim768_OVER.txt` |
| D4 dim=1024 (over) | `S10-D4_bench_dim1024_OVER.txt` |
| L2 10L | `S10-L2_bench_10L_dim512.txt` |
| L3 11L | `S10-L3_bench_11L_dim512.txt` |
| A2 MHA | `S10-A2_bench_mha_dim512.txt` |

### Session 11 — GPTQ tuning + ResFormer

| Run | Log filename |
|-----|-------------|
| Q1 sequential (failed) | `S11-Q1_gptq_sequential_FAILED.txt` |
| Q3 embed (failed) | `S11-Q3_gptq_embed_FAILED.txt` |
| Q7 all (failed) | `S11-Q7_gptq_all_FAILED.txt` |
| R0 α=0.0 | `S11-R0_resformer_a0.txt` |
| R1 α=0.1 | `S11-R1_resformer_a01.txt` |
| R3 α=0.5 (best) | `S11-R3_resformer_a05.txt` |
| R4 α=0.7 | `S11-R4_resformer_a07.txt` |

### Session 12 — V2 factorial F1–F9

| Run | Log filename |
|-----|-------------|
| F1 PR ctrl | `F1_v2_pr_none.txt` |
| F2 PR + headwise | `F2_v2_pr_head.txt` |
| F3 PR + elem (over) | `F3_v2_pr_elem_OVER.txt` |
| F4 RF ctrl | `F4_v2_rf_none.txt` |
| F5 RF + headwise | `F5_v2_rf_head.txt` |
| F6 RF + elem (over) | `F6_v2_rf_elem_OVER.txt` |
| F7 PR+RF | `F7_v2_both_none.txt` |
| F8 PR+RF + headwise | `F8_v2_both_head.txt` |
| F9 PR+RF + elem (over) | `F9_v2_both_elem_OVER.txt` |

### Session 13 — V2 compression C1–C7 (C5, C8 absent in logs)

| Run | Log filename |
|-----|-------------|
| C1 emb7 | `C1_v2_f2_emb7.txt` |
| C2 emb6 | `C2_v2_f2_emb6.txt` |
| C3 clip10 (over) | `C3_v2_f2_clip10_OVER.txt` |
| C4 clip8 (over) | `C4_v2_f2_clip8_OVER.txt` |
| **C6 official** | `C6_v2_f2_emb7_eclip15.txt` |
| C7 emb7+clip10+eclip15 (over) | `C7_v2_f2_emb7_clip10_eclip15_OVER.txt` |

### Session 15 — C6 fine-tuning sweep

| Run | Log filename |
|-----|-------------|
| E1 EMA=0.995 (best S15) | `S15-E1_c6_ema995.txt` |
| E2 EMA=0.997 | `S15-E2_c6_ema997.txt` |
| E3 EMA=0.999 (failed) | `S15-E3_c6_ema999_FAILED.txt` |
| W1 wd=0.08 | `S15-W1_c6_wd08.txt` |
| W2 wd=0.10 | `S15-W2_c6_wd10.txt` |
| W3 wd=0.11 | `S15-W3_c6_wd11.txt` |
| D1 warmdown=0.80 | `S15-D1_c6_warmdown80.txt` |
| Q1 qkg=5.5 | `S15-Q1_c6_qkg55.txt` |
| Q2 qkg=5.75 | `S15-Q2_c6_qkg575.txt` |
| Q3 qkg=6.0 | `S15-Q3_c6_qkg60.txt` |
| T1–T9 TTT epochs × LR | `S15-T1_..` through `S15-T9_c6_ttt_e7_lr1.txt` (T8 = best) |
| P0 paper baseline | `S15-P0_v2_p16p5_r0_baseline.txt` |
| P1–P3 LR warmup 2/5/10% | `S15-P1_..warmup002`, `S15-P2_..warmup005`, `S15-P3w_..warmup010` |
| P4–P5 Structured FFN | `S15-P4sffn_v2_sffn_r50_b4.txt`, `S15-P5_v2_sffn_r75_b8.txt` |
| Peri-LN (failed) | `S15-PeriLN_v2_p22p15_r1_FAILED.txt` |
| B2 small batch ga=1 | `B2_v2_p22p15_r2_ga1.txt` |
| B3 ga=1 + β2=0.99 | `B3_v2_p22p15_r3_ga1_b299.txt` |

### Session 16 — EMA deeper + PreQuantTTT

| Run | Log filename |
|-----|-------------|
| R1 EMA=0.995 + wd=0.10 | `S16-R1_c6_ema995_wd10.txt` |
| R2 EMA=0.993 | `S16-R2_c6_ema993.txt` |
| R3 EMA=0.990 (best EMA) | `S16-R3_c6_ema990.txt` |
| R4 PreQuantTTT (brotli, **C3 violation**) | `S16-R4_c6_prequant_ttt.txt` |
| R4 PreQuantTTT (lrzip, C3 violation) | `S16-R4lrzip_c6_prequant_lrzip_emb7.txt` |

### SLM (Sessions 6 INVALID + 7 valid)

| Run | Log filename |
|-----|-------------|
| Session 6 (SLM code absent) | `INVALID_S6_slm_test_k60.txt`, `INVALID_S6_slm_k{40,50,70,80,90}.txt`, `INVALID_S6_leaky_relu2_slm_k60.txt`, `INVALID_S6_leaky_relu2_headwise_slm_k60.txt`, `INVALID_S6_headwise_qkgain5_slm_k60.txt`, `INVALID_S6_sp8192_combo_slim_slm.txt` |
| S7-S1 SP1024 k=0.6 | `S7-S1_slm_val_sp1024_k60.txt` |
| S7-S2 SP8192 k=0.6 | `S7-S2_slm_val_sp8192_k60.txt` |
| S7-S3 SP8192 k=0.7 | `S7-S3_slm_val_sp8192_k70.txt` |
| S7-S4 SP8192 k=0.8 | `S7-S4_slm_val_sp8192_k80.txt` |

### X1 + V2-base + exploratory

| Run | Log filename |
|-----|-------------|
| X1 fullstack (PreQuantTTT, C3 violation) | `X1_fullstack_seed42_C3VIOL.txt`, `X1_fullstack_seed1337_C3VIOL.txt`, `X1_fullstack_seed2025_C3VIOL.txt` |
| V2 base reference | `V2-base_seed42.txt` |
| V2 C6 nogptqtune (incomplete) | `V2-c6_nogptqtune_INCOMPLETE.txt` |
| Early MQA / 1-GPU exploration | `S-explore_1gpu.txt`, `S-explore_mqa_1gpu.txt`, `S-explore_mqa_test.txt` |

### Inflated / duplicate (off the critical path)

| Note | Log filename |
|------|-------------|
| Byte-identical duplicate of `P3-fix_corrected_seed42.txt` (safe to `git rm`) | `_DUP_p3_corrected_seed42.txt` |
| Pre-fix P4 attempt with `caseops_enabled=False` (inflated bytes, ~0.97 BPB artifact) | `P4-pre_caseops_seed1337_INFLATED.txt` |
| P4 exploration with `caseops_enabled=False` + emb=8 (inflated bytes) | `P4-explore_caseops_off_emb8_INFLATED.txt` |
