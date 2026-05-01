# Experiment Configs Index

43 `.env` config files for Parameter Golf experiments (CS 7643).
Source with `source runs/configs/<name>.env` before running `torchrun`.

**Important:** Every config must explicitly set ALL toggles to prevent stale env var contamination between runs.

---

## Infrastructure / Templates (4)

| Config | Description |
|--------|-------------|
| `explore_1gpu.env` | 1 GPU exploration template, SP1024, 10 min |
| `explore_2gpu.env` | 2 GPU exploration template, SP1024, 10 min |
| `competition_8gpu.env` | 8 GPU competition template, SP1024, 10 min |
| `smoke_test.env` | Quick sanity check |

## V1 SP1024 Technique Experiments (5)

Single-technique ablations on the V1 (train_gpt.py) SP1024 baseline.

| Config | Technique | Best BPB (2xH100) |
|--------|-----------|-------------------|
| `gated_attn_elementwise.env` | Elementwise gated attention | 1.2653 |
| `gated_attn_headwise.env` | Headwise gated attention | 1.2659 |
| `leaky_relu2.env` | LeakyReLU squared | 1.2641 |
| `leaky_relu2_headwise.env` | LeakyReLU squared + headwise | 1.2642 |
| `headwise_qkgain5.env` | Headwise + QK-Gain 5.0 | — |

## V1 SP8192 Experiments (12)

SP8192 vocab + various architecture/compression combos on V1.

| Config | Key Variations |
|--------|---------------|
| `sp8192_combo.env` | Full SP8192 combo (dim=512, elementwise) |
| `sp8192_combo_slim.env` | Slim variant (dim=448, headwise) |
| `sp8192_combo_slim_gptq.env` | Slim + GPTQ compression |
| `sp8192_combo_slim_nottt.env` | Slim without TTT (ablation) |
| `sp8192_elementwise_dim416.env` | Elementwise, dim=416 |
| `sp8192_elementwise_dim448.env` | Elementwise, dim=448 |
| `sp8192_elementwise_gptq_dim512.env` | Elementwise + GPTQ, dim=512 |
| `sp8192_elementwise_int6_dim448.env` | Elementwise, int6, dim=448 |
| `sp8192_elementwise_int6_dim480.env` | Elementwise, int6, dim=480 |
| `sp8192_elementwise_int6_dim512.env` | Elementwise, int6, dim=512 |
| `sp8192_mqa_dim448.env` | MQA, dim=448 |
| `sp8192_mqa_elementwise_dim416.env` | MQA + elementwise, dim=416 |

## V2 Base + Benchmarks (10)

V2 = fork of rank 1's train_gpt_v2.py (11L x 512d, FA3, XSA, depth recurrence).

| Config | Description |
|--------|-------------|
| `v2_base.env` | Rank 1 defaults + our novelty toggles (off) |
| `v2_fullstack.env` | V2 with all our additions enabled |
| `gptq_tune_10L_mha.env` | GPTQ tuning base (dim=512, 10L, MHA) |
| `bench_dim448_elem.env` | Benchmark: dim=448, elementwise |
| `bench_dim512_elem.env` | Benchmark: dim=512, elementwise |
| `bench_dim768_elem.env` | Benchmark: dim=768, elementwise |
| `bench_dim1024_elem.env` | Benchmark: dim=1024, elementwise |
| `bench_10L_dim512_elem.env` | Benchmark: 10 layers, dim=512 |
| `bench_11L_dim512_elem.env` | Benchmark: 11 layers, dim=512 |
| `bench_mha_dim512_elem.env` | Benchmark: MHA (no GQA), dim=512 |

## Legal Submission Tuning (3)

Hyperparameter tuning for legal (non-PreQuantTTT) submissions.

| Config | Description |
|--------|-------------|
| `legal_ema090.env` | EMA decay=0.990 |
| `legal_ema090_smallbatch.env` | EMA=0.990 + small batch |
| `legal_ema090_clipsigma.env` | EMA=0.990 + clip sigma tuning |

## P1/P2 Phase (4)

Incremental improvements toward SOTA on PR #1851 base.

| Config | Description |
|--------|-------------|
| `p1a_hparam_sota.env` | Phase 1a: hyperparam SOTA attempt |
| `p1b_hparam_sota_loop3.env` | Phase 1b: + loop3 depth recurrence |
| `p2_smear_lqer.env` | Phase 2: SMEAR + LQER compression |
| `p2_lora_caseops.env` | Phase 2: LoRA + CaseOps tokenizer |

## P3 Phase (4)

PR #1851 base + our 3 novel contributions (headwise gate, EMA=0.990, small batch).

| Config | Description | Result |
|--------|-------------|--------|
| `p3_1851_headwise_ema_smallbatch.env` | P3 seed 42 (original, CASEOPS_ENABLED=0) | **RETRACTED** (1.0066 inflated) |
| `p3_1851_seed1337.env` | P3 seed 1337 | **RETRACTED** |
| `p3_1851_seed2025.env` | P3 seed 2025 | **RETRACTED** |
| `p3_caseops_corrected.env` | P3 corrected (CASEOPS_ENABLED=1, explicit paths) | 1.0972 BPB (seed 42) |

---

## Key Findings

- **Best legal submission:** C6 at 1.0805 BPB (3-seed mean, 8xH100)
- **P3 retracted:** byte accounting error from CASEOPS_ENABLED=0 + CaseOps tokenizer symlink inflated denominator by ~9%
- **LeakyReLU squared:** best free technique on SP1024 (+0.0008 BPB, zero cost)
- **EMA=0.990:** best hyperparameter finding (-0.012 BPB vs default 0.9965)
- **Small batch:** biggest V2 technique win (-0.015 BPB on 2xH100)

See `docs/parameter-golf/findings.md` for full results.
