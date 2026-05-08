# Log Renaming Plan — `parameter-golf/logs/`

Goal: rename every log so the filename starts with the **run label used in `findings.md`**, making cross-references in the final report trivial.

Scope: 135 `.txt` files in `parameter-golf/logs/` (excluding `requirements.txt`).

---

## Naming Convention

```
<RunLabel>_<descriptive_slug>.txt
```

- **Run label** matches `findings.md` exactly (zero-padded for numbered runs to keep `ls` ordering: `Run01`, `Run02`, ...).
- **Slug** preserves the existing descriptive name so `grep` keeps working.
- For letter-range collisions (e.g. Session 8 `E1` Elementwise vs Session 15 `E1` EMA vs Session 11 `Q0` GPTQ-tune vs Session 15 `Q1` QK-Gain), the slug disambiguates — no actual filename collision.
- `INVALID_` prefix preserved for invalid runs (kept first so they sort to top).
- `_DUP` suffix on byte-identical duplicates (recommend `git rm` after report is done).
- Caps suffix `_OVER` on over-budget runs and `_FAILED`/`_RETRACTED` for invalid/retracted results, so they're visually distinct in `ls`.

### Label dictionary (from findings.md)

| Letter | Sessions | Meaning |
|--------|----------|---------|
| Run01-13 | 1-7 | Numbered V1 runs (SP1024 + SP8192 baseline) |
| RunA / RunD / RunH | 6 | SP8192 combo slim 2×H100 reproducibility |
| F1-F9 | 12 | V2 3×3 factorial (PR/RF × None/Headwise/Elementwise) |
| C1-C8 | 13 | V2 compression knob sweep on F2 base |
| A1-A3 | 14 | 8×H100 ablation (F1 ctrl / F7 PR+RF / F2 headwise) |
| S1-S4 | 7 | SLM ratio sweep (real SLM, post-fix) |
| T1-T9 | 15 | TTT epochs × LR grid |
| Q1-Q3 | 15 | QK-Gain sweep on C6 |
| Q0-Q7 | 11 | GPTQ tuning experiments |
| E1-E3 | 15 | EMA sweep on C6 |
| E1-E4 | 8 | Elementwise + MQA sweep at SP8192 (different E1!) |
| W1-W3 | 15 | Weight-decay sweep |
| D1 | 15 | Warmdown sweep |
| R0-R4 | 11 | ResFormer α sweep |
| R1-R3 | 16 | EMA deeper sweep (R1=0.995+wd0.10, R2=0.993, R3=0.990) |
| R4-R5 | 16 | PreQuantTTT |
| P0-P5 | 15 | Paper #16 (LR Warmup) + Paper #5 (Structured FFN) |
| P1a / P1b / P1c | 18 / 22 | SOTA hparam adoption (P1c = clip12) |
| P2 | unused | (config exists, no log) |
| P3 | 19 | PR #1851 fork + EMA=0.990 + small batch (RETRACTED) |
| P4 / P4b | 22 | CaseOps + SOTA hparams (P4b = NEW BEST 1.0621) |
| L1-L2 | TBD | Legal EMA=0.990 on 8×H100 (`legal_ema090*`) |
| B2-B3 | 15 | Small Batch ga=1 (Paper #15) |
| N1-N2 | TBD | DiffAttn / Cross-Seq Attn (Sessions ~17-18) |
| X1 | unsubmitted | Fullstack with PreQuantTTT (C3 violation) |
| Sxx_ prefix | varies | Session marker for variants without a unique label |

---

## Section 1 — Submittable / Pinned Runs (high-priority)

These appear in the official 8×H100 leaderboard table at the top of findings.md.

| Current | Proposed | Run label / Notes |
|---|---|---|
| `p4b_caseops_seed42.txt` | `P4b_caseops_seed42.txt` | **NEW BEST 1.0621** (May 5, 350 KB log) |
| `v2_c6_seed42.txt` | `C6_v2_c6_seed42.txt` | C6 3-seed (1.0818) |
| `v2_c6_seed1337.txt` | `C6_v2_c6_seed1337.txt` | C6 3-seed (1.0794, best of 3) |
| `v2_c6_seed2025.txt` | `C6_v2_c6_seed2025.txt` | C6 3-seed (1.0804) |
| `v2_c6_nogptqtune_8gpu.txt` | `A1_v2_f1_8gpu.txt` | A1 control on 8×H100 |
| `v2_f1_8gpu.txt` | `A1_v2_f1_8gpu_alt.txt` | Possible duplicate of A1; verify before merge |
| `v2_f7_8gpu.txt` | `A2_v2_f7_8gpu.txt` | A2 ablation (PR+RF, α=0.5) |
| `p1a_hparam_sota.txt` | `P1a_hparam_sota.txt` | P1a (1.0769, over budget) |
| `p1a_hparam_sota_seed42.txt` | `P1a_hparam_sota_seed42_DUP.txt` | Likely duplicate, verify |
| `p1b_hparam_sota_loop3.txt` | `P1b_hparam_sota_loop3.txt` | P1b (1.0793, NUM_LOOPS=3 hurts) |
| `p1b_hparam_sota_loop3_seed42.txt` | `P1b_hparam_sota_loop3_seed42_DUP.txt` | Likely duplicate, verify |
| `p3_1851_seed42_emb6.txt` | `P3_inflated_seed42_RETRACTED.txt` | P3 inflated bytes (1.0069) |
| `p3_1851_seed1337_emb6.txt` | `P3_inflated_seed1337_RETRACTED.txt` | P3 inflated bytes (1.0057) |
| `p3_1851_seed2025_emb6.txt` | `P3_inflated_seed2025_RETRACTED.txt` | P3 inflated bytes (1.0073) |
| `p3_corrected_seed42.txt` | `P3-fix_corrected_seed42.txt` | Corrected (1.0972) |
| `p3_corrected_seed1337.txt` | `P3-fix_corrected_seed1337.txt` | Corrected (1.0978) |
| `p3_corrected_seed42_incomplete.txt` | `P3-fix_corrected_seed42_incomplete.txt` | Truncated rerun |
| `p4_clip12.txt` | `P1c_p4_clip12.txt` | Session 22 P1c (MATRIX_CLIP_SIGMAS=12.0) |
| `legal_ema090.txt` | `L1_legal_ema090.txt` | L1 EMA=0.990 8×H100 (1.0830) |
| `legal_ema090_seed42.txt` | `L1_legal_ema090_seed42_DUP.txt` | Likely duplicate, verify |
| `legal_ema090_smallbatch.txt` | `L2_legal_ema090_smallbatch.txt` | L2 EMA=0.990 + small batch (1.0926) |
| `legal_ema090_smallbatch_seed42.txt` | `L2_legal_ema090_smallbatch_seed42_DUP.txt` | Likely duplicate, verify |

---

## Section 2 — V1 Numbered Runs (Sessions 1-7)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `baseline_sp1024.txt` | `Run01_baseline_sp1024.txt` | Run 1 (1.3045, 1×GPU bug) |
| `gated_attn_headwise.txt` | `Run02_gated_attn_headwise.txt` | Run 2 (1.2653) |
| `gated_attn_elementwise.txt` | `Run03_gated_attn_elementwise_OVER.txt` | Run 3 (1.2602, 17.87 MB) |
| `mqa.txt` | `Run04_mqa_OVER.txt` | Run 4 (1.2761, 16.84 MB) |
| `gqa_baseline.txt` | `Run05_INVALID_stale_env.txt` | Run 5 INVALID (215 KB log, 19.4M params from stale env) |
| `gqa_baseline_v2.txt` | `Run06_gqa_baseline.txt` | Run 6 clean baseline (1.2667) |
| `leaky_relu2.txt` | `Run07_leaky_relu2.txt` | Run 7 (1.2641) |
| `leaky_relu2_headwise.txt` | `Run08_leaky_relu2_headwise.txt` | Run 8 (1.2642) |
| `headwise_qkgain5.txt` | `Run09_headwise_qkgain5.txt` | Run 9 (1.2719) |
| `sp8192_combo.txt` | `Run10_sp8192_combo_OVER.txt` | Run 10 (1.1872, 19.41 MB) |
| `sp8192_combo_slim.txt` | `Run11_sp8192_combo_slim.txt` | Run 11 (1.2077, first submittable) |
| `sp8192_combo_slim_8gpu.txt` | `Run11_sp8192_combo_slim_8gpu_DUP.txt` | Likely duplicate of Run 11; verify |
| `sp8192_combo_slim_2gpu.txt` | `RunA_sp8192_combo_slim_2gpu.txt` | Run A (1.2411) |
| `sp8192_combo_slim_2gpu_v2.txt` | `RunD_sp8192_combo_slim_2gpu.txt` | Run D (1.2396, A repeat) — verify |
| `sp8192_combo_slim_nottt_2gpu.txt` | `RunH_sp8192_combo_slim_nottt.txt` | Run H (1.2432, no TTT) |
| `sp8192_combo_slim_seed42.txt` | `Run11-3seed_sp8192_combo_slim_seed42.txt` | 3-seed reproducibility (V1) |
| `sp8192_combo_slim_seed1337.txt` | `Run11-3seed_sp8192_combo_slim_seed1337.txt` | 3-seed reproducibility |
| `sp8192_combo_slim_seed2025.txt` | `Run11-3seed_sp8192_combo_slim_seed2025.txt` | 3-seed reproducibility |
| `headwise_qkgain5_2gpu.txt` | `Run09-rerun_headwise_qkgain5_2gpu.txt` | Run 9 2-GPU retest (1.2981) |
| `leaky_relu2_2gpu.txt` | `Run07-rerun_leaky_relu2_2gpu.txt` | Run 7 2-GPU retest |

---

## Section 3 — Session 8 (Elementwise + MQA Sweep, SP8192)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `sp8192_elem_int6_dim512_2gpu.txt` | `S8-E1_elem_dim512_OVER.txt` | E1 (1.2338, over budget) |
| `sp8192_elem_int6_dim448_2gpu.txt` | `S8-E2_elem_dim448.txt` | E2 (1.2447) |
| `sp8192_elem_int6_dim480_2gpu.txt` | `S8-extra_elem_dim480.txt` | dim=480 (not in findings table) |

---

## Section 4 — Session 9 (GPTQ Validation)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `gptq_bench_int6_ar.txt` | `S9-G3_gptq_int6_ar.txt` | G3 (1.3081 TTT) |
| `gptq_bench_int6_train.txt` | `S9-G4_gptq_int6_train.txt` | G4 (1.3030) |
| `gptq_bench_int7_train.txt` | `S9-G2_gptq_int7_train.txt` | G2 (1.2924, BEST GPTQ) |

---

## Section 5 — Session 10 (Benchmark Sweep)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `bench_dim448.txt` | `S10-D1_bench_dim448.txt` | D1 (1.2859) |
| `bench_dim512.txt` | `S10-D2_bench_dim512.txt` | D2 (1.2684) |
| `bench_dim768.txt` | `S10-D3_bench_dim768_OVER.txt` | D3 (1.2287, 27.14 MB) |
| `bench_dim1024.txt` | `S10-D4_bench_dim1024_OVER.txt` | D4 (1.2127, 46.47 MB) |
| `bench_10L_dim512.txt` | `S10-L2_bench_10L_dim512.txt` | L2 (1.2664) |
| `bench_11L_dim512.txt` | `S10-L3_bench_11L_dim512.txt` | L3 (1.2591) |
| `bench_mha_dim512.txt` | `S10-A2_bench_mha_dim512.txt` | A2 (1.2577, MHA wins) |

---

## Section 6 — Session 11 (GPTQ Tuning + ResFormer)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `gptq_tune_sequential.txt` | `S11-Q1_gptq_sequential_FAILED.txt` | Q1 (1.3916, +0.13 worse) |
| `gptq_tune_embed.txt` | `S11-Q3_gptq_embed_FAILED.txt` | Q3 (1.6897, catastrophic) |
| `gptq_tune_all.txt` | `S11-Q7_gptq_all_FAILED.txt` | Q7 (1.8679, worst) |
| `resformer_a0.txt` | `S11-R0_resformer_a0.txt` | R0 (1.2584, control) |
| `resformer_a01.txt` | `S11-R1_resformer_a01.txt` | R1 (1.2545) |
| `resformer_a05.txt` | `S11-R3_resformer_a05.txt` | R3 (1.2536, BEST α=0.5) |
| `resformer_a07.txt` | `S11-R4_resformer_a07.txt` | R4 (1.2551) |

---

## Section 7 — Session 12 (V2 Factorial F1-F9)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `v2_pr_none.txt` | `F1_v2_pr_none.txt` | F1 PR ctrl (1.1641) |
| `v2_pr_head.txt` | `F2_v2_pr_head.txt` | F2 PR+headwise (1.1636) |
| `v2_pr_elem.txt` | `F3_v2_pr_elem_OVER.txt` | F3 PR+elem (1.1665, 17.21 MB) |
| `v2_rf_none.txt` | `F4_v2_rf_none.txt` | F4 RF ctrl (1.1666) |
| `v2_rf_head.txt` | `F5_v2_rf_head.txt` | F5 RF+headwise (1.1661) |
| `v2_rf_elem.txt` | `F6_v2_rf_elem_OVER.txt` | F6 RF+elem (1.1700) |
| `v2_both_none.txt` | `F7_v2_both_none.txt` | F7 PR+RF (1.1636, tied best) |
| `v2_both_head.txt` | `F8_v2_both_head.txt` | F8 PR+RF+headwise (1.1650) |
| `v2_both_elem.txt` | `F9_v2_both_elem_OVER.txt` | F9 PR+RF+elem (1.1686) |

---

## Section 8 — Session 13 (V2 Compression C1-C8)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `v2_f2_emb7.txt` | `C1_v2_f2_emb7.txt` | C1 emb7 (1.1656) |
| `v2_f2_emb6.txt` | `C2_v2_f2_emb6.txt` | C2 emb6 (1.1735) |
| `v2_f2_clip10.txt` | `C3_v2_f2_clip10_OVER.txt` | C3 clip10 (1.1605, 17.31 MB) |
| `v2_f2_clip8.txt` | `C4_v2_f2_clip8_OVER.txt` | C4 clip8 (1.1598, 18.67 MB) |
| `v2_f2_emb7_clip10_eclip15.txt` | `C7_v2_f2_emb7_clip10_eclip15_OVER.txt` | C7 (1.1596, 17.01 MB) |
| `v2_f2_emb7_eclip15.txt` | `C6_v2_f2_emb7_eclip15.txt` | **C6 OFFICIAL** (1.1622) |

(Note: C5 and C8 don't appear to have logs in this directory — verify with user.)

---

## Section 9 — Session 15 (C6 Fine-Tuning Sweep)

### EMA + WD (Sessions 15 + 16 share E* / R* labels — disambiguated by slug)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `c6_ema995.txt` | `S15-E1_c6_ema995.txt` | E1 (1.1562, **best Session 15**) |
| `c6_ema997.txt` | `S15-E2_c6_ema997.txt` | E2 (1.1690) |
| `c6_ema999.txt` | `S15-E3_c6_ema999_FAILED.txt` | E3 (1.3475, catastrophic) |
| `c6_wd08.txt` | `S15-W1_c6_wd08.txt` | W1 (1.1650) |
| `c6_wd10.txt` | `S15-W2_c6_wd10.txt` | W2 (1.1619) |
| `c6_wd11.txt` | `S15-W3_c6_wd11.txt` | W3 (1.1628) |
| `c6_warmdown80.txt` | `S15-D1_c6_warmdown80.txt` | D1 (1.1645) |
| `c6_qkg55.txt` | `S15-Q1_c6_qkg55.txt` | Q1 (1.1642) |
| `c6_qkg575.txt` | `S15-Q2_c6_qkg575.txt` | Q2 (1.1634) |
| `c6_qkg60.txt` | `S15-Q3_c6_qkg60.txt` | Q3 (1.1636) |

### TTT epochs × LR (T1-T9)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `c6_ttt_e3_lr3.txt` | `S15-T1_c6_ttt_e3_lr3.txt` | T1 (1.1665) |
| `c6_ttt_e3_lr5.txt` | `S15-T2_c6_ttt_e3_lr5.txt` | T2 (1.1650) |
| `c6_ttt_e3_lr1.txt` | `S15-T3_c6_ttt_e3_lr1.txt` | T3 (1.1633) |
| `c6_ttt_e5_lr3.txt` | `S15-T4_c6_ttt_e5_lr3.txt` | T4 (1.1643) |
| `c6_ttt_e5_lr5.txt` | `S15-T5_c6_ttt_e5_lr5.txt` | T5 (1.1636) |
| `c6_ttt_e5_lr1.txt` | `S15-T6_c6_ttt_e5_lr1.txt` | T6 (1.1634) |
| `c6_ttt_e7_lr3.txt` | `S15-T7_c6_ttt_e7_lr3.txt` | T7 (1.1631) |
| `c6_ttt_e7_lr5.txt` | `S15-T8_c6_ttt_e7_lr5.txt` | T8 (1.1624, **best TTT**) |
| `c6_ttt_e7_lr1.txt` | `S15-T9_c6_ttt_e7_lr1.txt` | T9 (1.1629) |

### Paper experiments P0-P5 (LR Warmup + Structured FFN)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `v2_p16p5_r0_baseline.txt` | `S15-P0_v2_p16p5_r0_baseline.txt` | P0 baseline (1.1572) |
| `v2_p16p5_r1_warmup002.txt` | `S15-P1_v2_p16p5_r1_warmup002.txt` | P1 warmup 2% (1.1596) |
| `v2_p16p5_r2_warmup005.txt` | `S15-P2_v2_p16p5_r2_warmup005.txt` | P2 warmup 5% (1.1614) |
| `v2_p16p5_r3_warmup010.txt` | `S15-P3-w_v2_p16p5_r3_warmup010.txt` | P3-warmup (1.1638) — note P3 collision with submission P3 |
| `v2_p16p5_r4_sffn_r50_b4.txt` | `S15-P4-sffn_v2_sffn_r50_b4.txt` | P4-sffn (1.1997) — note P4 collision |
| `v2_p16p5_r5_sffn_r75_b8.txt` | `S15-P5_v2_sffn_r75_b8.txt` | P5 (1.2068) |

### Paper #15 (Small Batch) + Peri-LN

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `v2_p22p15_r1_peri_ln.txt` | `S15-PeriLN_v2_p22p15_r1_FAILED.txt` | Peri-LN (NaN/1.8369) |
| `v2_p22p15_r2_ga1.txt` | `B2_v2_p22p15_r2_ga1.txt` | B2 small batch (1.1419) |
| `v2_p22p15_r3_ga1_b299.txt` | `B3_v2_p22p15_r3_ga1_b299.txt` | B3 small batch + β2=0.99 (1.1422) |

---

## Section 10 — Session 16 (EMA Deeper + PreQuantTTT)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `c6_ema995_wd10.txt` | `S16-R1_c6_ema995_wd10.txt` | R1 (1.1559) |
| `c6_ema993.txt` | `S16-R2_c6_ema993.txt` | R2 (1.1526) |
| `c6_ema990.txt` | `S16-R3_c6_ema990.txt` | R3 (1.1505, **best EMA**) |
| `c6_prequant_ttt.txt` | `S16-R4_c6_prequant_ttt.txt` | R4 PreQuantTTT (1.0507, NON-COMPLIANT) |
| `c6_prequant_lrzip_emb7.txt` | `S16-R4-lrzip_c6_prequant_lrzip_emb7.txt` | R4 lrzip variant |

---

## Section 11 — SLM (Sessions 6 INVALID + Session 7 valid)

### Session 6 — INVALID (SLM code absent on pod)

| Current | Proposed | Notes |
|---|---|---|
| `INVALID_slm_test_k60_2gpu.txt` | `INVALID_S6_slm_test_k60.txt` | |
| `INVALID_slm_sweep_k40_2gpu.txt` | `INVALID_S6_slm_k40.txt` | |
| `INVALID_slm_sweep_k50_2gpu.txt` | `INVALID_S6_slm_k50.txt` | |
| `INVALID_slm_sweep_k70_2gpu.txt` | `INVALID_S6_slm_k70.txt` | |
| `INVALID_slm_sweep_k80_2gpu.txt` | `INVALID_S6_slm_k80.txt` | |
| `INVALID_slm_sweep_k90_2gpu.txt` | `INVALID_S6_slm_k90.txt` | |
| `INVALID_leaky_relu2_slm_k60_2gpu.txt` | `INVALID_S6_leaky_relu2_slm_k60.txt` | |
| `INVALID_leaky_relu2_headwise_slm_k60_2gpu.txt` | `INVALID_S6_leaky_relu2_headwise_slm_k60.txt` | |
| `INVALID_headwise_qkgain5_slm_k60_2gpu.txt` | `INVALID_S6_headwise_qkgain5_slm_k60.txt` | |
| `INVALID_sp8192_combo_slim_slm_2gpu.txt` | `INVALID_S6_sp8192_combo_slim_slm.txt` | |

### Session 7 — Real SLM (S1-S4)

| Current | Proposed | Run label / val_bpb |
|---|---|---|
| `slm_val_s1_sp1024_k60.txt` | `S7-S1_slm_val_sp1024_k60.txt` | S1 (1.4204) |
| `slm_val_s2_sp8192_k60.txt` | `S7-S2_slm_val_sp8192_k60.txt` | S2 (1.4034) |
| `slm_val_s3_sp8192_k70.txt` | `S7-S3_slm_val_sp8192_k70.txt` | S3 (1.3201) |
| `slm_val_s4_sp8192_k80.txt` | `S7-S4_slm_val_sp8192_k80.txt` | S4 (1.2652) |

---

## Section 12 — Other / X1 / Exploratory

| Current | Proposed | Notes |
|---|---|---|
| `x1_fullstack_seed42.txt` | `X1_fullstack_seed42_C3VIOL.txt` | X1 (1.0517, PreQuantTTT C3 violation) |
| `x1_fullstack_seed1337.txt` | `X1_fullstack_seed1337_C3VIOL.txt` | X1 (1.0513) |
| `x1_fullstack_seed2025.txt` | `X1_fullstack_seed2025_C3VIOL.txt` | X1 (1.0502) |
| `v2_base.txt` | `V2-base_seed42.txt` | V2 base reference (~1.05, may also be inflated bytes) |
| `v2_c6_nogptqtune.txt` | `V2-c6_nogptqtune_INCOMPLETE.txt` | Incomplete/diverged log (val_bpb=3.48) |
| `mqa_test.txt` | `S-explore_mqa_test.txt` | Early MQA exploration |
| `mqa_1gpu.txt` | `S-explore_mqa_1gpu.txt` | Early MQA exploration |
| `explore_1gpu.txt` | `S-explore_1gpu.txt` | Early exploration |
| `requirements.txt` | *(keep)* | Not a log |

---

## Section 13 — UUID / Duplicate Files (require user decision)

| Current | Proposed | Decision required |
|---|---|---|
| `0d6ec472-4cba-46b7-afe4-2ed64633ead3 (1).txt` | `_DUP_p3_corrected_seed42.txt` or **delete** | Same size as `p3_corrected_seed42.txt` (175268 B) — almost certainly a true duplicate. Recommend `git rm`. |
| `56425ee1-5461-4a73-9216-3c5ff401da55.txt` | `P4-explore_caseops_off_emb8_INFLATED.txt` | Unique config: `caseops_enabled=False, embed_bits=8, ema=0.9965`, val_bpb 0.9756. Suspected inflated-byte-denom artifact like P3. Doesn't match any findings.md run label. Suggest renaming with `_INFLATED` suffix and adding to findings as a footnote. |
| `p4_caseops_seed1337 (1).txt` | `P4-pre_caseops_seed1337_INFLATED.txt` | val_bpb 0.9757 (also suspected inflated bytes). Different from P4b (1.0621). Pre-fix P4 attempt? |

---

## Section 14 — Files Possibly Missing from Mapping

The following config files exist but I couldn't find a matching log — verify they were never run, or they live elsewhere:

- `bench_11L_mha_dim512_elem.env` (no `bench_11L_mha_dim512.txt` log)
- `p2_lora_caseops.env`, `p2_smear_lqer.env` (no P2 logs)
- `p3_caseops_corrected.env` (covered by `p3_corrected_*.txt`)
- `smoke_test.env` (smoke tests may not be saved)
- `gated_attn_headwise.txt` is 335 KB — much larger than other 2-GPU logs — may concatenate multiple runs. Worth a quick `grep "run_id:"` count.

---

## Execution Plan

After your review:

1. **Use `git mv`** for every rename — preserves git history (these logs are tracked).
2. **Group renames by section** so each commit is atomic and easy to revert (one commit per section).
3. **Don't delete duplicates yet** — keep `_DUP` suffix until report is finalized.
4. **Verify `gated_attn_headwise.txt` size** before renaming (might be multiple concatenated runs).

I'll dump a `rename.sh` script with all `git mv` commands once you approve.

---

## Open Questions for User

1. **C5 / C8** — no logs found. Do they exist elsewhere or were they never run?
2. **`gated_attn_headwise.txt` 335 KB** — concatenated logs? Want me to split?
3. **`56425ee1-...` and `p4_caseops_seed1337 (1).txt`** — both look like pre-P4b experiments with `caseops_enabled=False` (inflated bytes). Add a footnote to findings.md, or just rename and move on?
4. **Letter prefix scheme**: `S15-T8_c6_ttt_e7_lr5.txt` vs simpler `T8_c6_ttt_e7_lr5.txt`. The session-prefix avoids letter-range collisions but adds noise. Preference?
5. **`legal_ema090*` files** — pairs of `*.txt` and `*_seed42.txt` with same val_bpb (1.0830 / 1.0926). These look like the L1/L2 runs but with confusing seed-suffix duplicates. Want me to inspect each pair before committing renames?
