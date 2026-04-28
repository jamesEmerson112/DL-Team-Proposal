# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Deep Learning Team Proposal** repository for CS 7643. The project investigates efficient language model training under extreme compression constraints through **OpenAI's Parameter Golf** challenge — training the best model within a 16 MB artifact budget and 10-minute wall clock on 8×H100 GPUs. We study which architectural and optimization techniques (SP8192 vocab, LeakyReLU², gated attention, QK-Gain, Score-First TTT, SLM) contribute most to model quality. 30+ experiments completed, best submittable result: **1.2077 BPB** (Run 11).

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
- Best submittable result: **1.2077 BPB** (Run 11, SP8192 combo slim + TTT) — beats PG baseline (1.2244)
- 3-seed reproducibility confirmed: mean 1.2073 ±0.0006 BPB
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
