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
- 35+ experiments completed across 2×H100 and 8×H100, 21 experiment configs, 6 run scripts
- Best submittable result: **1.0805 BPB** (C6, V2 headwise + emb7+eclip15, 3-seed mean on 8×H100) — matches SOTA (1.0810)
- 3-seed reproducibility confirmed: C6 mean 1.0805 ±0.0012 BPB (V1 Run 11: 1.2073 ±0.0006)
- **DO NOT SUBMIT YET** — matches SOTA but doesn't clear >=0.005 nats threshold for SOTA record. Keep technique secret.
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

### 2026-04-29 (Session 15)
- [run] Executed 19-run C6 fine-tuning sweep on 2×H100 (`run_v2_finetune_2gpu.sh`):
  - TTT Grid (T1-T9): 9 runs, {3,5,7} epochs × {0.003, 0.005, 0.01} LR — none beat baseline. T8 (e7, lr0.005) = 1.1624, +0.0002 vs C6. Default (e3, lr0.005) is near-optimal.
  - QK-Gain (Q1-Q3): 5.5/5.75/6.0 — all worse. Rank 1 default (5.25) optimal.
  - **EMA (E1-E3): E1 (decay=0.995) = 1.1562, -0.0060 vs C6 — BIG WIN.** E2 (0.997) = 1.1690 worse. E3 (0.999) = 1.3475 catastrophic.
  - WD (W1-W3): W2 (wd=0.10) = 1.1619, -0.0003 marginal.
  - Warmdown (D1): frac=0.80 = 1.1645, worse than default 0.72.
- [finding] **EMA=0.995 is the single best hyperparameter finding** — more aggressive weight averaging helps at this training duration
- [finding] EMA sensitivity extreme: 0.995 (best) → 0.9965 (default) → 0.997 (worse) → 0.999 (catastrophic)
- [finding] Projected 8×H100 with EMA=0.995: ~1.0745 BPB (from C6 1.0805 - 0.006)
- [edit] Updated findings.md with full 19-run table, added E1 and W2 to 2×H100 leaderboard
- [edit] Budget check added: train_gpt_v2.py = 50 KB, total submission ~15.74 MB, under budget

### 2026-04-30 (Session 16)
- [research] Checked PG leaderboard PRs — **SOTA exploded to 1.0136 BPB** (PR #1958, okezue)
  - Key new techniques: PreQuantTTT (21ep AdamW on val before GPTQ, ~0.06 BPB), sliding-window stride-64 eval (~0.01 BPB), per-group lrzip compression, LQER, CaseOps tokenizer
  - Multiple PRs in 1.05-1.07 range. Our C6 (1.0805) now ranks ~9th-10th.
- [edit] Updated findings.md leaderboard: new SOTA 1.0136, added top 8 new PRs, updated submission strategy
- [edit] Updated neurlps-paper-survey.md: marked papers #19, #21, #22, #23, #12 as "paper read"
- [feat] Created `docs/James_notes/TODO.txt` — added Priority 1 (HybridNorm, Schedule-Free, DiffAttn) and Priority 2 (Peri-LN, Structured FFN, MATES) training techniques
- [feat] Implemented PreQuantTTT in `parameter-golf/train_gpt_v2.py`:
  - New `prequant_ttt()` function: 21 epochs AdamW, freezes blocks 0-1 + embeddings, cosine LR 5e-4→5e-5, federated avg across GPUs
  - New env vars: `PREQUANT_TTT_ENABLED`, `PREQUANT_TTT_EPOCHS`, `PREQUANT_TTT_LR`, `PREQUANT_TTT_LR_END`
  - Wired into `train_and_eval()` between post-EMA eval and GPTQ serialization
- [feat] Implemented per-group compression (`COMPRESSOR=pergroup`): pure Python lzma/brotli hybrid, picks smaller per 64KB chunk. No system deps.
- [feat] Created `runs/run_v2_session16_2gpu.sh` — unified 6-run Phase 1-3 script:
  - Phase 1: EMA deeper sweep (0.995+WD0.10, 0.993, 0.990) — auto-picks best
  - Phase 2: PreQuantTTT with best EMA
  - Phase 3: Per-group compression (emb7 + emb8 variants)
- [feat] Updated `runs/configs/v2_base.env` with PreQuantTTT defaults (disabled by default)
- [edit] Updated `docs/James_test/run_finetune_2gpu_commands.txt` for Session 16
- [todo] Run Session 16 on 2×H100, then 8×H100 3-seed if results are competitive (~1.015 projected)

### 2026-04-30 (Session 16 — Run Results)
- [run] Executed 5-run Session 16 sweep on 2×H100 (`run_v2_session16_2gpu.sh`):
  - Phase 1 EMA Deeper Sweep (R1-R3): EMA=0.990 is new best (R3, TTT 1.1505, -0.0117 vs C6). EMA=0.993 (R2, 1.1526) and EMA=0.995+WD0.10 (R1, 1.1559) also beat E1 (1.1562). EMA keeps improving as decay decreases.
  - Phase 2-3 PreQuantTTT (R4-R5): R4 (brotli) = **1.0507 TTT BPB** on 2×H100 — beats our 8×H100 C6 (1.0805). PreQuantTTT takes pre-Q 1.1591 → post-PQ 1.0156 (-0.1435 BPB). R5 (pergroup) crashed at deserialize.
- [bugfix] Fixed `deserialize()` crash in `train_gpt_v2.py`: `torch.load()` needs `weights_only=False` for PyTorch 2.11+ (default changed to `True`)
- [finding] **EMA=0.990 is optimal** — nearly 2× the gain of 0.995 (-0.0117 vs -0.0060 below C6). Projected 8×H100: ~1.069 BPB
- [finding] **PreQuantTTT is transformative** — 21 epochs AdamW on val before GPTQ gives -0.1435 BPB. Single biggest technique gain found in entire project
- [finding] **2×H100 R4 (1.0507) beats 8×H100 C6 (1.0805)** — PreQuantTTT more impactful than 4× GPU scaling
- [finding] Projected 8×H100 with EMA=0.990 + PreQuantTTT: ~0.97-1.00 BPB — would beat SOTA (1.0136)
- [edit] Updated `docs/parameter-golf/findings.md`: added Session 16 section, R1-R4 to 2×H100 leaderboard, updated "Where We Stand"
- [todo] Run 8×H100 3-seed with EMA=0.990 + PreQuantTTT — projected to beat SOTA

### 2026-04-30 (Session 15-16 — Paper Experiments on James-experiment-2)
- [branch] Created `James-experiment-2` from `James-experiment` for Paper #16/#5/#22/#15 experiments
- [feat] Implemented LR Warmup (`LR_WARMUP_FRAC` env var) in `parameter-golf/train_gpt_v2.py` — modifies `lr_mul()` to add linear ramp before warmdown
- [feat] Implemented Structured FFN (`StructuredMLP` class) — low-rank up-proj + block-diagonal down-proj with `STRUCTURED_FFN`, `FFN_RANK_RATIO`, `FFN_NUM_BLOCKS` env vars
- [feat] Implemented Peri-LN (`PERI_LN` env var) — output RMSNorm on attn + MLP in Block.forward
- [feat] Made `grad_accum_steps` configurable via `GRAD_ACCUM_STEPS` env var (was hardcoded `8//world_size`)
- [run] Paper #16 (LR Warmup): tested 2%, 5%, 10% on 2×H100. **All FAILED** — monotonically worse (+0.0024 to +0.0066 BPB). Rank 1 was right to skip warmup.
- [run] Paper #5 (Structured FFN): tested r=0.5/b=4 and r=0.75/b=8 on 2×H100. **FAILED** — saves 30-56% MLP params but +0.04-0.05 BPB degradation. Paper tested at 125M+; doesn't transfer to 36M.
- [run] Paper #22 (Peri-LN): **FAILED** — immediate NaN. Output norms destabilize rank 1 stack (conflicts with attn_scale/mlp_scale + ln_scale_factor).
- [run] Paper #15 (Small Batch): tested ga=1 + TRAIN_BATCH_TOKENS=196608 on 2×H100. **SUCCESS: -0.015 BPB** (B2=1.1419 vs baseline 1.1572). Biggest V2 technique win. 3,349 steps vs ~1,030. Beta2 scaling (0.99) makes no difference.
- [bugfix] Fixed OOM in small batch runs — must reduce TRAIN_BATCH_TOKENS by 4x when setting GRAD_ACCUM_STEPS=1
- [bugfix] Fixed `set -uo pipefail` crash in run script — `$BETA2` unset, used `${BETA2:-0.95}` fallback
- [feat] Created run scripts: `runs/run_v2_paper16_paper5_2gpu.sh`, `runs/run_v2_paper22_paper15_2gpu.sh`
- [feat] Created 8 new env configs for all experiments
- [feat] Created runbooks: `docs/James_test/run_paper16_paper5_2gpu_commands.txt`, `docs/James_test/run_paper22_paper15_2gpu_commands.txt`
- [feat] Created `docs/James_test/small_batch_merge_guide.txt` — 2-edit guide for porting Small Batch to James-experiment branch
- [edit] Updated `docs/parameter-golf/findings.md` — added B2/B3 (small batch), P0-P5 (LR warmup + structured FFN) to tables, resolved 3 merge conflicts, added key insights #11-15
- [edit] Updated `docs/parameter-golf/neurlps-paper-survey.md` — marked Papers #5, #16 as "Tested — FAILS"
- [merge] Applied Small Batch edits to `James-experiment` branch (2 edits to train_gpt_v2.py)
- [finding] PG challenge deadline: April 30, 2026 at 4:59 PM PST
- [finding] Small Batch is the only technique that improved on the V2 rank 1 stack. Papers #1 (SLM), #5 (Structured FFN), #16 (LR Warmup), #20 (ResFormer), #22 (Peri-LN) all failed.
- [todo] Test Small Batch on 8×H100 — on 8 GPUs ga is already 1, need to reduce TRAIN_BATCH_TOKENS to get more updates. Projected: ~1.0655 BPB if -0.015 delta holds.

### 2026-04-30 (Session 17 — C6 Legal Submission)
- [feat] Updated `docs/James_notes/submission.json` — C6 3-seed data (mean 1.0805 BPB, std 0.0012), removed PreQuantTTT, set `no_pre_quant_ttt: true`, removed Small Batch/EMA tuning references (not used in C6), removed "rank 1 SOTA" from bigbag attribution (no longer rank 1)
- [feat] Rewrote `docs/James_notes/pg_submission_readme.md` — C6 values throughout, marked as non-record submission, removed PreQuantTTT/Small Batch/EMA=0.990 sections (C6 uses default EMA 0.9965 and default batch 786432), updated compliance to "No Pre-Quantization TTT — fully legal", updated reproduction command (removed PREQUANT_TTT_ENABLED, EMA_DECAY, GRAD_ACCUM_STEPS, TRAIN_BATCH_TOKENS), cleaned credits (removed @okezue PreQuantTTT attribution)
- [feat] Updated pg-fork (`jamesEmerson112/parameter-golf`): renamed submission folder from `2026-04-30_SP8192_FullStack_HeadwiseGate_PreQuantTTT` to `2026-04-30_SP8192_HeadwiseGate_EMA_LegalTTT`, replaced all 7 files (clean train_gpt.py with 0 prequant references, C6 seed logs, updated JSON/README)
- [feat] Force-pushed to `submission/fullstack-headwise-gate` branch
- [feat] Created new PR #2005 on openai/parameter-golf (PR #1992 was CLOSED, so created fresh): "Record: SP8192 + Headwise Gated Attention + Legal TTT (1.0805 BPB, 3-seed)"
- [finding] C6 config confirmed from logs: gated_attn=headwise, embed_bits=7, embed_clip_sigmas=15.0, ema_decay=0.9965 (default), train_batch_tokens=786432 (default), grad_accum_steps=1 (8 GPUs), no PreQuantTTT
- [finding] C6 verified results: seed42=1.0818, seed1337=1.0794, seed2025=1.0804, mean=1.0805 ±0.0012 BPB
- [finding] bigbag is no longer rank 1 — updated all references to remove "rank 1 SOTA"
- [ref] PR: https://github.com/openai/parameter-golf/pull/2005
- [ref] pg-fork clone at /tmp/pg-fork (ephemeral, not persisted locally)

### 2026-05-01 (Session 20)
- [edit] Updated `docs/parameter-golf/findings.md` with P3 SOTA results:
  - Added P3 row to 8×H100 table (1.0066 BPB, 12,382 steps, ~15.97 MB)
  - Updated SOTA references: 1.0066 (us, P3, PR #2071), previous 1.0611 (codemath3000)
  - Added P3 3-seed reproducibility table (mean 1.0066 ±0.0009)
  - Added Session 19 — P3 SOTA Runs (config table, 3-seed results, key findings, compliance)
  - Restructured Official Leaderboard table — P3 at top as NEW
  - Updated Key Insights #8 ($1,165/130+), #10 (P3 IS SOTA), added #20-22
- [edit] Updated `docs/parameter-golf/neurlps-paper-survey.md`: header, leaderboard table, Paper #15, Techniques in Use — all reflect P3 1.0066 SOTA
- [edit] Updated `docs/James_notes/pr2005_supplement.md`: EMA/Small Batch/Transfer sections with P3 data, fixed all "rank 1" → "@bigbag" (7 occurrences), experiment scale 130+/$1,165
- [finding] **CaseOps data was actually used in P3** — pod had symlinks pointing standard SP8192 paths to CaseOps-tokenized data, even though `CASEOPS_ENABLED=0` was set. Updated all docs + pg-fork PR to reflect "CaseOps ON (via symlinked data)"
- [feat] Updated pg-fork PR #2071: fixed CaseOps status, added full reproduction steps (download from romeerp/parameter-golf-caseops-v1 + symlink commands), updated PR description and acknowledgements
- [finding] P3 logs ARE complete (4,112-4,120 lines per seed) — reviewer saw GitHub diff view truncation, not actual truncation
- [research] Reviewed competing PRs: #2098 (0.80051, PPM+TTT), #2083 (0.94175, PPM no TTT), #2066 (SSM hybrid, non-record) — PPM submissions likely C2 violations per PR #1905 mathematical proof
- [finding] P3 is the strongest pure neural network submission — no PPM, no byte mixing, clean C2 compliance
- [ref] PR #2071: https://github.com/openai/parameter-golf/pull/2071

### 2026-05-01 (Session 21 — P3 Byte Accounting Retraction)
- [research] Investigated reviewer's byte accounting concern for PR #2071 (P3 submission). Traced full code path: `CASEOPS_ENABLED=0` + CaseOps tokenizer via symlink → `build_sentencepiece_luts()` at line 474 → LUT byte counting in `_accumulate_bpb()` at line 2691 → inflated byte denominator (~164.6M CaseOps-transformed bytes vs ~151M canonical raw bytes)
- [finding] **P3 RETRACTED — byte accounting error confirmed.** Original 1.0066 BPB was artifact of inflated byte denominator. Rerun with `CASEOPS_ENABLED=1` (sidecar byte counting) gives **1.0972 BPB** (seed 42) — worse than C6 (1.0805) by +0.0167. val_loss (~2.401) was identical, confirming the issue was purely the byte denominator, not model quality.
- [finding] Root cause: on the pod, standard paths were symlinked to CaseOps data (`ln -s fineweb10B_sp8192_lossless_caps_caseops_v1_reserved data/datasets/fineweb10B_sp8192`). With `CASEOPS_ENABLED=0`, code loaded CaseOps tokenizer but skipped sidecar file, using LUT byte counting that includes case marker bytes not present in raw text.
- [finding] Confirmed sidecar file exists in `romeerp/parameter-golf-caseops-v1` HF repo: `datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin` (1 sidecar for 1 val shard)
- [finding] Download script (`data/cached_challenge_fineweb.py`) has a gap — it downloads token shards but NOT `fineweb_val_bytes_*.bin` sidecar files. Must download sidecar manually.
- [finding] Default `_default_caseops_data` path in `train_gpt.py` (line 381-388) doesn't match where the download script puts data. Must set `DATA_PATH` explicitly when using `CASEOPS_ENABLED=1`.
- [bugfix] Permission error during GPTQ calibration: `fineweb_train_000067.bin` had restrictive permissions from HF cache hard-links. Fixed with `chmod +r`.
- [feat] Created `docs/James_test/run_p3_rerun_correct_bytes.txt` — runbook for rerunning P3 with correct byte accounting (CASEOPS_ENABLED=1 + explicit DATA_PATH + sidecar download)
- [edit] Major update to `docs/parameter-golf/findings.md`: P3 retracted throughout — 8×H100 table, summary, compliance note, 3-seed table, Session 19 findings, legality feedback (added rerun results), leaderboard, Where We Stand, Key Insights #20/#22/#23. Best legal run reverted to C6 (1.0805).
- [finding] **Best legal run: C6 at 1.0805 BPB** (3-seed mean). Gap to external SOTA (1.0611, PR #1855): +0.0194. P3's techniques (EMA=0.990 + small batch) do NOT help on PR #1851 stack at 8×H100 — consistent with L1/L2 failures on @bigbag stack.
- [finding] CaseOps byte accounting is a silent trap — `CASEOPS_ENABLED=0` with CaseOps tokenizer inflates byte denominator by ~9% without warning. Must use `CASEOPS_ENABLED=1` with sidecar for correct BPB.
- [ref] Rerun runbook: `docs/James_test/run_p3_rerun_correct_bytes.txt`

### 2026-05-01 (Session 22 — P3 Corrected Logs + P1c Planning)
- [feat] Read and analyzed 3 corrected P3 log files (CASEOPS_ENABLED=1, sidecar byte counting): `0d6ec472` (seed 42), `9a386796` (seed 1337), `7a3f8113` (seed 42, incomplete/truncated)
- [finding] Corrected P3 2-seed results: seed42=1.09724, seed1337=1.09779, mean=1.0975 BPB — confirms +0.0170 worse than C6 (1.0805)
- [finding] Corrected P3 steps: ~12,140 (due to small batch 196K tokens), but sees FEWER total tokens (~2.4B) than C6 (~3.5B with default 786K batch). More steps ≠ more learning.
- [finding] PR #1851 base uses default batch (~4,930 steps). The ~12K steps are purely from our small batch override.
- [feat] Renamed UUID log files: `0d6ec472...` → `p3_corrected_seed42.txt`, `9a386796...` → `p3_corrected_seed1337.txt`, `7a3f8113...` → `p3_corrected_seed42_incomplete.txt`
- [edit] Updated `docs/James_notes/p1a_vs_p3_comparison.txt`: added run descriptions, P1c column to config table, 2-seed P3 reproducibility section, corrected steps/BPB/size
- [feat] Created `runs/configs/p4_clip12.env` — P1c config: P1a + MATRIX_CLIP_SIGMAS=12.0 (tightened from 11.5 to fit under 16 MB budget)
- [edit] Updated `docs/James_test/run_x1_8gpu_commands.txt` — clean pod instructions for P1c run (no comments, just commands)
- [finding] MATRIX_CLIP_SIGMAS confirmed at `parameter-golf/train_gpt.py:320` — reads from env with default 12.85
- [todo] Run P1c on 8×H100 pod. Expected: BPB ~1.077-1.080, size <16 MB. If passes → 3-seed for new best legal submission.
