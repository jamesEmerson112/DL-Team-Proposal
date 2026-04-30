# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Deep Learning Team Proposal** repository for CS 7643. The project investigates efficient language model training under extreme compression constraints through **OpenAI's Parameter Golf** challenge — training the best model within a 16 MB artifact budget and 10-minute wall clock on 8×H100 GPUs. We study which architectural and optimization techniques (SP8192 vocab, LeakyReLU², gated attention, QK-Gain, Score-First TTT, SLM) contribute most to model quality. 35+ experiments completed, best submittable result: **1.0805 BPB** (C6, 3-seed mean on 8×H100 — matches SOTA).

## Repository Structure

- `parameter-golf/` — Forked competition repo (submodule, contains `train_gpt.py` with our modifications)
- `runs/` — Experiment launch scripts + `configs/` (21 .env config files)
- `docs/James_test/` — Research notes (numbered 00-20, plus compute-plan.md)
- `docs/parameter-golf/` — Competition overview, findings, paper survey (29 papers)
- `tools/` — Plotting utilities (`plot_curves.py`)
- `README.md` — Project overview
- `LICENSE` — MIT License

**Note:** `docs/James_notes/` is gitignored (private working copies). `docs/James_test/` is the committed notes folder.

## Current State

Active research + experimentation repository. Code modifications in `parameter-golf/train_gpt.py` include SLM (Selective Language Modeling), LeakyReLU² activation, gated attention (headwise/elementwise), QK-Gain scaling, and Score-First TTT. Project focus has shifted fully to the **Parameter Golf pipeline** — nanochat research is background context only.

## Key Context

- **Parameter Golf** is the active competition target: 16 MB artifact, 10 min on 8×H100, scored by FineWeb validation BPB
- 30+ experiments completed across 2×H100 and 8×H100, 21 experiment configs, 6 run scripts
- Best submittable result: **1.0805 BPB** (C6, V2 headwise + emb7+eclip15, 3-seed mean on 8×H100) — matches SOTA (1.0810)
- 3-seed reproducibility confirmed: C6 mean 1.0805 ±0.0012 BPB (V1 Run 11: 1.2073 ±0.0006)
- **DO NOT SUBMIT YET** — matches SOTA but doesn't clear ≥0.005 nats threshold for SOTA record. Keep technique secret.
- nanochat is the official successor to nanoGPT (released Oct 2025) — studied for technique porting but not used directly

## Context History

### 2026-04-15
- Read all project files (README, CLAUDE.md, docs/ notes 00-12, training-plan, experiments tracker, findings, proposal requirements, parameter-golf overview, proposal-draft) to map full project state
- Created ASCII visualization showing knowledge-vs-execution gap: 12 research notes (~95% done) but training plan, experiments, findings, and submission all at 0-10%
- Identified 5 blocking decisions the team needs to resolve before proceeding
- [feat] Created `docs/dashboard.html` — self-contained dark-themed HTML dashboard (629 lines) with: two-track overview (CS 7643 + OpenAI Parameter Golf), 7 progress bars with hover tooltips, 3-phase flow diagram with task checklists, baseline run results card (d=3, val_bpb=1.160, CORE=0.036), 5 blocking decisions panel, Gantt timeline (Apr 15-30) with live "today" marker, knowledge-vs-execution gap chart
- Dashboard uses GitHub-dark theme, CSS Grid/Flexbox, JS for live countdown and BPB scale
- [finding] Project has excellent research foundations but nearly zero execution artifacts — one baseline run (d=3) completed out of ~12+ planned experiments
- [todo] Parameter Golf deadline is April 30, 2026

### 2026-04-16 (Session 2)
- Ran Parameter Golf baseline on RunPod pod (2x GPU auto-detected, intended 1 GPU)
- Encountered disk quota exceeded error during FineWeb sp1024 dataset download — resolved by deleting HF cache after download (`rm -rf /workspace/.cache/huggingface/hub/datasets--willdepueoai--parameter-golf`)
- Training result: val_bpb 1.3045 (baseline to beat: 1.2244, gap: +0.0801)
- Model: 17M params, 9 layers, 512 dims, GQA, 1024 vocab
- Completed 1,819/20,000 steps before hitting 10-min wall clock cap — model still improving
- int8+zlib compressed model: 14.7 MB (under 16 MB budget), compression nearly lossless (+0.001 BPB)
- Total cost: ~$2 for < 1 hour of pod time
- Logged full run details to `docs/parameter-golf/findings.md` (Run 1 section)
- Key insight: 8 GPUs should process ~4x more tokens in same 10-min window, likely beating 1.2244 baseline
- [todo] Next run: use 8xH100 to maximize throughput within wall clock cap

### 2026-04-19
- [decision] Teammate raised concern that nanochat vs Parameter Golf integration is "impossible" — concluded it's hard but unnecessary; no need to fork nanochat's codebase
- [decision] Chose Path A: use PG's pipeline directly (`train_gpt.py`), study nanochat's techniques (RoPE, RMSNorm, ReLU², Muon optimizer, GQA, depth-scaling), and port them one at a time into PG's code
- [research] PG submission process: fork repo → add folder under `records/track_10min_16mb/` → PR back to `openai/parameter-golf`. 28 record submissions exist, current SOTA is 1.0810 BPB (baseline 1.2244)
- [feat] Installed GitHub CLI (`gh`) on Windows, authenticated as `jamesEmerson112`
- [feat] Forked `openai/parameter-golf` → `github.com/jamesEmerson112/parameter-golf`
- [feat] Cloned fork to `C:\Users\voan2\Documents\GitHub\parameter-golf` (sibling to DL-Team-Proposal)
- Confirmed `runs/parameter_golf_baseline.sh` auto-discovers the sibling `parameter-golf/` directory — no script updates needed
- Experiment workflow confirmed: edit `train_gpt.py` → run with `torchrun` → read `val_bpb` from output → compare to baseline (1.2244) and SOTA (1.0810)

### 2026-04-23 (Session 3)
- Implemented LeakyReLU² activation support in `parameter-golf/train_gpt.py`: added `ACTIVATION` env var (default "relu2", option "leaky_relu2" = LeakyReLU(0.5)²), threaded through MLP → Block → GPT → instantiation site
- Created experiment configs: `runs/configs/leaky_relu2.env` and `runs/configs/leaky_relu2_headwise.env`
- [bug] Caught critical stale env var bug: Run 5 (intended as clean GQA baseline) produced 19.4M params instead of 17M because shell had leftover `GATED_ATTN=elementwise` from a previous `source runs/configs/gated_attn_elementwise.env`. The gated attention code is correct (conditionally allocates gate dims), but env vars persist across runs in the same shell session.
- [fix] Added explicit `GATED_ATTN=none` and `ACTIVATION=relu2` defaults to ALL env configs (explore_1gpu, explore_2gpu, competition_8gpu, smoke_test, gated_attn_elementwise, gated_attn_headwise, leaky_relu2, leaky_relu2_headwise) so sourcing any config always resets both toggles — prevents stale env var contamination
- [fix] Updated budget check in `runs/parameter_golf_baseline.sh` to use int8+zlib compressed artifact size (`*.ptz`) instead of raw `.pt` files — matches actual PG submission format
- Logged Run 5 in `docs/parameter-golf/findings.md` as INVALID (stale env var), noted root cause and fix
- [insight] When using `source` to load env configs, variables persist in the shell — every config must explicitly set ALL experiment toggles, not just the ones it changes
- Ran 4 experiments on 2×H100 pod (PyTorch 2.11, 10-min wall clock):
  - Run 6 + 6v2: Clean GQA baseline — 17M params confirmed, val_bpb 1.2649-1.2667, under 16 MB budget
  - Run 7: LeakyReLU² — **best technique**, val_bpb 1.2641, free (no extra params, no speed cost)
  - Run 8: LeakyReLU² + headwise — val_bpb 1.2642, combo doesn't stack (headwise adds speed penalty without quality gain on top of LeakyReLU²)
- [feat] Added copy-paste summary block to `runs/parameter_golf_baseline.sh` — auto-extracts run_id, params, val_loss, val_bpb, size, budget status from log file
- [fix] Fixed summary script bugs: warmup_step regex collision with step count, step_avg grabbing step-0 value, val_bpb_raw picking up roundtrip value instead of last training val
- [feat] Added val_loss (raw + int8+zlib) to all run entries and comparison tables in `docs/parameter-golf/findings.md`
- [feat] Added summary table at top of `docs/parameter-golf/findings.md` — all runs sorted by BPB
- [feat] Created `runs/configs/headwise_qkgain5.env` — QK-Gain 5.0 + headwise gated attn (PG ranks 1-6 all use 5.0-5.25). Still pending run.
- Marked headwise/elementwise gated attention as James Vo's original technique in findings
- [finding] LeakyReLU² is the best legal technique: +0.0008 BPB over baseline, zero cost. Headwise alone is competitive but doesn't stack with LeakyReLU². Gap to PG baseline (1.2244) still +0.0397.
- [todo] Run headwise_qkgain5.env — could be a big mover based on leaderboard

### 2026-04-23 (Session 4)
- [research] Searched 10 topic areas beyond existing 18-paper NeurIPS survey to find new techniques for Parameter Golf (17M param GPT, 16MB, 10min 8xH100)
- Found 12 new applicable techniques; expanded `docs/parameter-golf/neurips-paper-survey.md` from 18 to 29 papers
- Merged `docs/parameter-golf/paper-survey.md` into `docs/parameter-golf/neurips-paper-survey.md` (deleted duplicate)
- Added "PG Leaderboard" column to all 29 papers indicating which are proven on the competition leaderboard
- Expanded Top 5 actionable techniques to Top 8, added "Leaderboard-Proven Techniques" section (7 paper-backed + 8 competition techniques)
- Added "Papers Blocked/Low Priority" section for techniques that don't fit PG constraints
- Key new papers found:
  - Value Residual Learning / ResFormer (ACL 2025) — 16% fewer params equivalent, on leaderboard rank 8
  - Differential Attention (ICLR 2025 Oral) — two-softmax-subtract attention, noise cancellation
  - HybridNorm (NeurIPS 2025) / Peri-LN (ICML 2025) — better norm placement, zero extra params
  - Schedule-Free Optimizer (NeurIPS 2024 Oral) — no LR schedule needed, ideal for fixed wall clock
  - Early Weight Averaging (COLM 2024) — proven on PG leaderboard ranks 7,12,13,14
  - FlashAttention-3 (NeurIPS 2024) — proven on leaderboard rank 15, 1.5-2x over FA2 on H100
  - Exclusive Self-Attention / XSA (arXiv 2026) — proven on leaderboard ranks 8,10,14,15
- [finding] PG SOTA updated: current best is ~1.028 BPB (down from 1.081 previously noted); top submissions now use LoRA TTT, cross-sequence attention, and pre-quantization TTT
- [todo] Prioritize leaderboard-proven techniques for next runs: EWA, FlashAttention-3, XSA, Value Residual Learning

### 2026-04-26 (Session 5)
- [research] Deep-dived into Rho-1 paper (NeurIPS 2024 Best Paper Runner-Up) — Selective Language Modeling (SLM). Explained token categories (H→H, H→L, L→H, L→L), excess loss formula, and how only ~26% of tokens drive meaningful learning
- [research] Explored XSA (Exclusive Self-Attention, arXiv 2026) and Differential Attention (ICLR 2025 Oral) — two attention modifications proven on PG leaderboard
- [research] Confirmed microsoft/rho GitHub repo has NO training code — only pretrained models + eval. Must implement SLM ourselves
- [feat] Created `docs/James_notes/19_rho1-selective-language-modeling.md` — comprehensive research notes with 4-phase testing plan (smoke test → ratio sweep → competition run → 3-seed submission), pass/fail criteria, estimated costs (~$26 total)
- [feat] Implemented SLM Option A (simple loss-threshold) in `parameter-golf/train_gpt.py`:
  - Added `SLM_ENABLED` + `SLM_RATIO` env vars to Hyperparameters class
  - Added params to `GPT.__init__` signature + model instantiation
  - Modified loss at line 743: when SLM enabled during training, uses `F.cross_entropy(reduction="none")` + `torch.topk` to keep top k% tokens by loss, averages only those. Validation still uses full mean loss.
- [feat] Updated ALL 11 existing .env configs with `SLM_ENABLED=0` + `SLM_RATIO=0.6` defaults (prevents stale env var contamination)
- [feat] Created 4 new experiment configs: `slm_test.env` (k=0.6), `slm_sweep_50.env`, `slm_sweep_70.env`, `slm_sweep_80.env`
- [feat] Updated `runs/parameter_golf_baseline.sh` summary block to print slm_enabled/slm_ratio
- [feat] Created `runs/parameter_golf_8gpu_3seed_run.sh` — wrapper that runs 3 seeds (42, 1337, 2025) and computes mean/std val_bpb with submission.json snippet
- [feat] Created `runs/run_all_2gpu.sh` — sequential runner for all 2xH100 experiments (SP8192 combo slim retry + SLM phases 1-3), with results summary table at the end
- [feat] Updated `docs/James_test/run12_2gpu_commands.txt` and `docs/James_test/run_8gpu_commands.txt` with SLM experiment commands (Runs A-E with notes and pass/fail criteria)
- [edit] Sorted paper survey table in `docs/parameter-golf/neurlps-paper-survey.md` by year (2026→2023), kept original # numbers for reference stability
- [note] Paper #15 (Small Batch Size Training) flagged as low-hanging fruit — just remove grad_accum_steps + tune beta2, no code change needed. Keeping in mind for later.
- [finding] TTT (Test-Time Training) is NOT from a paper in our survey — it's a practitioner-developed eval-time trick from the PG competition (attributed to @dexhunter PR #1413)

### 2026-04-26 (Session 6)
- [bugfix] Fixed NGPUS bug in `run_all_2gpu.sh` — config files were overwriting `NGPUS=2` to `NGPUS=1` via `source`. Fix: removed `export NGPUS=` from all 15 .env configs. Baseline scripts have safe fallbacks (`NGPUS="${NGPUS:-1}"`)
- [feat] Expanded `runs/run_all_2gpu.sh` from 6 to 14 runs (added E-L: headwise+QKG5, LeakyReLU², combos with SLM, SP8192 no-TTT ablation, extended SLM sweep k=0.4/0.9)
- [feat] Created 6 new .env configs: `leaky_relu2_slm.env`, `sp8192_combo_slim_nottt.env`, `slm_sweep_40.env`, `slm_sweep_90.env`, `headwise_qkgain5_slm.env`, `leaky_relu2_headwise_slm.env`
- [run] Executed 14-run sweep on 2×H100 + 1 baseline confirm + 1 money shot (16 total runs)
- [run] Baseline confirm: 1.2659 BPB, 171ms step_avg, 3,508 steps — matches old pods exactly
- [finding] SP8192 dominates: all SP8192 runs (1.238-1.243) beat every SP1024 run (1.289+) by ~0.05 BPB
- [finding] SLM k=0.8 is optimal ratio from sweep (k=0.4 to 0.9)
- [finding] SLM improves SP8192 combo: Run A (no SLM) 1.2411 → Run D (SLM k=0.6) 1.2396 → Run 13 (SLM k=0.8) **1.2384** — new best 2×H100 run
- [finding] TTT contributes ~0.002 BPB on 2×H100 (Run A 1.2411 vs Run H no-TTT 1.2432)
- [finding] Techniques stack on SP1024: L (LReLU²+headwise+SLM) 1.2899 > F (LReLU² only) 1.2932
- [finding] Projected 8×H100 with SLM k=0.8: ~1.2050 BPB (from Run 11's 1.2077 - 0.0027 SLM delta)
- [edit] Updated `docs/James_test/run12_2gpu_commands.txt` — replaced old runs A-D with single "money shot" run (SP8192 combo slim + SLM k=0.8)
- [edit] Added Run 13 (1.2384 BPB) to `docs/parameter-golf/findings.md` 2×H100 table
- [feat] Created `docs/James_test/pg_grant_application.txt` — PG Development grant ($500) application with 3 fields: approach (1,500 chars), tried so far (255 chars), PR link
- [ref] PR submission: https://github.com/openai/parameter-golf/pull/1799
- [user] User is An Thien Vo, Georgia Tech grad student, CS 7643 Deep Learning. Spent $240+ personal funds on PG experiments.

### 2026-04-27 (Session 9)
- [edit] Updated `docs/James_test/18_llm-parameter-anatomy.md` — expanded from SP1024 baseline-only to include 4 new sections:
  - **SP8192 Combo Slim (Run 11, 16.4M):** full param breakdown at dim=448 with headwise gate, comparison table vs baseline showing embedding growth (3.1%→22.4%) and dim-reduction tradeoffs
  - **SP8192 Combo (Run 10, 20.8M):** dim=512 param breakdown, delta table showing where +4.4M params come from (MLP half, attention third, embedding sixth), elementwise variant note
  - **Quantization Anatomy:** aggregate compression comparison (int8+zlib 15.35 MB vs GPTQ 10.50 MB), per-component sensitivity table (embedding=low tolerance, MLP=high, early blocks 30× more important than late), 4 GPTQ bugs/lessons (inference_mode poison, percentile vs k×std, int7>int6, train>AR calibration)
  - **Budget Math:** GPTQ compression ratio (0.684×) applied to 5 configs, showing dim=512/11L/MLP3×/elementwise all fit under 16 MB with GPTQ
- [edit] Updated cross-reference section with current `train_gpt.py` line numbers (GPT class moved 765→1370 after GPTQ additions)
- [preserved] Original PG Baseline (SP1024, 17M) section and Forward Pass Pipeline unchanged

### 2026-04-28 (Session 10)
- [feat] Created 7 benchmark env configs (`bench_dim448_elem.env`, `bench_dim512_elem.env`, `bench_dim768_elem.env`, `bench_dim1024_elem.env`, `bench_10L_dim512_elem.env`, `bench_11L_dim512_elem.env`, `bench_mha_dim512_elem.env`) and `runs/run_benchmark_sweep_2gpu.sh`
- [run] Executed 7-run sweep on 2×H100: dim (448/512/768/1024), layers (9/10/11), attention (GQA/MHA) — all elementwise + GPTQ int7 + train data calib
- [finding] Pre-quant BPB scales well with model size: D2=1.2120, L3=1.2042, A2=1.2045 — all beat baseline (1.2244) pre-quant
- [finding] GPTQ gap ~0.05 is the sole bottleneck — 4× worse than Kevin Clark's ~0.012
- [finding] Bigger models have smaller GPTQ gaps: D4 (dim=1024) gap=0.026, D1 (dim=448) gap=0.054
- [finding] Best under-budget candidates if GPTQ gap fixed: L3 (11L, 1.2042 pre-Q, 15.27 MB) and A2 (MHA, 1.2045 pre-Q, 14.24 MB)
- [finding] MHA beats GQA by -0.0107 TTT BPB at +1.3 MB cost (A2 vs D2)
- [edit] Updated `docs/parameter-golf/findings.md` with Session 10 benchmark sweep results
- [edit] Updated `docs/James_test/run12_2gpu_commands.txt` with benchmark sweep step

### 2026-04-28 (Session 11)
- [feat] Implemented 4 GPTQ tuning techniques: sequential block quantization (`GPTQ_SEQUENTIAL`), hook-based Hessian collection (`GPTQ_USE_HOOKS`), GPTQ on embeddings (`GPTQ_EMBED`), configurable dampening (`GPTQ_PERCDAMP`)
- [feat] Implemented ResFormer value residual learning (`VALUE_RESIDUAL_ALPHA`): v = (1-alpha)*v + alpha*v0, caches V from layer 0, blends into all subsequent layers
- [feat] Created `runs/run_combined_2gpu.sh` (GPTQ tuning + ResFormer in one script), `runs/run_gptq_tune_2gpu.sh`, `runs/run_gptq_tune_trimmed_2gpu.sh`, `runs/run_resformer_sweep_2gpu.sh`
- [feat] Created `runs/configs/gptq_tune_10L_mha.env` (dim=512, 10L, MHA base config for tuning)
- [run] GPTQ tuning Q0-Q7: Q0 baseline (1.2579 TTT), Q1 sequential (1.3916 — WORSE), Q3 embed GPTQ (1.6897 — WORSE), Q7 all combined (1.8679 — WORST)
- [finding] **GPTQ tuning approaches all failed** — sequential, embed GPTQ, and combined all dramatically worse than baseline. Sequential Hessians through dequantized blocks are inferior. Embedding GPTQ's frequency Hessian is wrong for lookup tables.
- [run] ResFormer alpha sweep R0-R4: R0=1.2584, R1(0.1)=1.2545, R3(0.5)=1.2536, R4(0.7)=1.2551
- [finding] **ResFormer works!** Alpha=0.5 is optimal: pre-Q 1.2004 (-0.0036), TTT 1.2536 (-0.0048 vs control). Zero extra params/size. GPTQ gap also slightly improved (0.0532 vs 0.0544).
- [edit] Updated `docs/parameter-golf/findings.md` with Session 11 results
- [bugfix] Fixed embedding GPTQ Hessian shape: was [vocab, vocab], needed [dim, dim]. Fixed with `H = W^T @ diag(freq) @ W`
- [bugfix] Fixed sequential GPTQ double-quantization: save/restore original weights, only use sequential Hessians

### 2026-04-28 (Session 12)
- [feat] Implemented V2 factorial plan: forked rank 1's train_gpt.py (1.0810 BPB) as `parameter-golf/train_gpt_v2.py`
  - Decompressed rank 1's LZMA wrapper (470 lines) into editable Python
  - Added gated attention (`GATED_ATTN` env var): Q projection widened by gate_dim, gate applied after FA3+XSA, before output proj
  - Added ResFormer value residual (`VALUE_RESIDUAL_ALPHA` env var): V₀ cached from first encoder layer, blended via `(1-alpha)*v + alpha*v0`
  - Adapted tensor shapes for FA3 output format `[bsz, seqlen, num_heads, head_dim]` (vs SDPA's `[bsz, num_heads, seqlen, head_dim]`)
  - Rank 1 stack preserved: FA3, 11L×512d, 4×MLP, LeakyReLU², depth recurrence (3-4-5 loop), parallel residuals, sigmoid skip gates, partial RoPE (16/64), XSA, MuonEq-R, EMA, GPTQ int6+brotli
- [feat] Created `runs/configs/v2_base.env` — rank 1 defaults with our novelty toggles (GATED_ATTN=none, VALUE_RESIDUAL_ALPHA=0.0)
- [fix] Removed no-op DATA_PATH/TOKENIZER_PATH from v2_base.env — rank 1 uses DATA_DIR, not these env vars
- [feat] Created `runs/run_v2_factorial_2gpu.sh` — 9-run 3×3 factorial sweep with results summary parser matching rank 1's log format
- [edit] Updated `docs/James_test/run12_2gpu_commands.txt` — added step 9 for V2 factorial sweep with FA3 install note
- [plan] 3×3 factorial: (PR only vs RF only vs PR+RF) × (No Gate vs Headwise vs Elementwise) = 9 runs
  - F1 = rank 1 control (no additions), F2-F3 = +gated attn, F4-F6 = ResFormer instead of PR, F7-F9 = both
  - Success criteria: any run beats our best 2×H100 BPB (1.2338)
- [todo] Run V2 factorial on 2×H100 RunPod (~117 min, 9 runs × ~13 min each). Requires FA3 wheel install.

### 2026-04-28 (Session 13)
- [research] Read "Selective Attention Improves Transformer" (Leviathan et al., ICLR 2025, arXiv:2410.02703) — parameter-free masking matrix subtracted from attention logits. Assessed as medium-low priority for PG: zero params but not proven on leaderboard, GQA untested, competes with XSA already in V2 stack. Different from Rho-1 SLM (attention routing vs gradient filtering).
- [feat] Created `docs/James_test/pg_rank1_slides.txt` — 3-slide presentation for team:
  - Slide 1: Architecture Innovations (6 techniques: depth recurrence, parallel residuals, skip gates, LeakyReLU², LN scale, partial RoPE)
  - Slide 2: Attention, Optimization & Compression (6 techniques: XSA, QK-Gain, MuonEq-R, EMA, GPTQ, TTT)
  - Slide 3: Full Stack synthesis (5 compound chains, key numbers, our results)
  - Each technique annotated with [Survey #N] or [Not in survey] cross-references
  - 9 paper references with arxiv links
- [edit] Updated `docs/parameter-golf/neurlps-paper-survey.md` — major refresh:
  - Header: "Our best" updated from 1.2653 to 1.2077 (Run 11) + V2 best 1.1636 (F2), date to 2026-04-28
  - ResFormer #20: updated to "Tested — CONTEXT-DEPENDENT" (works on simple stack, fails on rank 1 stack)
  - MoEUT #7: updated to "Proven + Used by us (paper read)"
  - Structured FFN #5: marked "paper read"
  - Paper #13 (Variable Seq Length): marked "may not be helpful"
  - Paper #16 (LR Warmup): marked "needs double-check"
  - Leaderboard table: Gated Attention updated to 1.1636 (V2 F2), added "Full rank 1 stack" row
  - EMA #25 and Depth Recurrence #7 in Top 8: marked "NOW IN USE"
  - "Techniques Already in Use" section: expanded from 7 items to V1 stack (9 items) + V2 stack (+12 items) with [Survey #N] tags
  - XSA and NorMuon struck through in "Techniques to Investigate" (now in use)
  - Zero stale "1.2653" references remain

### 2026-04-29 (Session 14)
- [run] Executed C6 submission + ablation on 8×H100 RunPod (PyTorch 2.11, CUDA 13.0, FA3)
- [run] Part A: 3-seed C6 (headwise + emb7+eclip15) — seeds 42/1337/2025
  - S1 (seed 42): **1.0818 BPB**, 15,697,552 bytes, 4,469 steps, eval 394s
  - S2 (seed 1337): **1.0794 BPB**, 15,694,065 bytes, 4,465 steps, eval 335s
  - S3 (seed 2025): **1.0804 BPB**, 15,693,855 bytes, 4,467 steps, eval 334s
  - Mean: **1.0805 BPB** (std ±0.0012) — matches SOTA (1.0810)
- [run] Part B: Ablation (seed 42, all 3 completed)
  - A1 (F1 control): **1.0806 BPB**, 15,977,755 bytes
  - A2 (F7 PR+RF α=0.5): **1.0828 BPB**, 15,983,964 bytes
  - A3 (F2 headwise, default compression): **1.0801 BPB**, 15,993,169 bytes (total 16,043,196 — over budget)
- [finding] **Headwise gate helps at 8×H100** — A3 (1.0801) beats A1 (1.0806) by −0.0005 BPB, consistent with 2×H100.
- [finding] **Compression tuning costs +0.0017 BPB** — C6 (1.0818) vs A3 (1.0801). emb7+eclip15 trades quality for size.
- [finding] **ResFormer hurts at 8×H100 scale** — A2 (α=0.5) 1.0828, worst of the four.
- [finding] **We match SOTA** (1.0805 vs 1.0810) but don't clear ≥0.005 nats threshold for SOTA record.
- [decision] **DO NOT SUBMIT** — keep headwise gated attention technique secret until gap is widened.
- [feat] Created `runs/run_v2_c6_8gpu.sh` — automated 6-run script (3 seeds + 3 ablations)
- [feat] Rewrote `docs/James_test/run_8gpu_commands.txt` — setup + calls run script
- [feat] Created `docs/James_test/pg_submission_guide.txt` — full PR submission process guide
- [feat] Created `docs/James_test/submission.json` — filled template with 8×H100 results
- [edit] Updated `docs/parameter-golf/findings.md` — added V2 C6 8×H100 results, 3-seed table, ablation table, submission strategy, Session 14 log, updated leaderboard position and key insights
