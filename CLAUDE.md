# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Deep Learning Team Proposal** repository. The project objective is to **train a language model using nanochat, then create multiple model variants to produce a learning curve comparison** studying which techniques (e.g., SFT, RLHF, depth/hyperparameter configurations) contribute most to model improvement. Background research also covers **nanoGPT** and **llm.c** (all by Andrej Karpathy) for comparative context.

## Repository Structure

- `docs/James_notes/` — James's research notes (numbered 01-07, with 00 as table of contents)
- `README.md` — Project overview
- `LICENSE` — MIT License

## Current State

This is a **research-only repository** — no code implementation. Work products are documentation and comparative analysis of the three frameworks being evaluated:

| Framework | Language | Scope | Status |
|-----------|----------|-------|--------|
| nanoGPT | Python/PyTorch | Pretraining only | Deprecated |
| llm.c | C/CUDA | Pretraining only | Active |
| nanochat | Python/PyTorch | Full pipeline (pretrain + SFT + RLHF + chat UI) | Active, successor to nanoGPT |

## Key Context

- nanochat is the official successor to nanoGPT (released Oct 2025)
- The team has an existing comparison table covering language, dependencies, performance, parameters, training stages, cost, and ease of use across all three frameworks
- llm.c offers ~7% faster performance than PyTorch with minimal dependencies (pure C/CUDA)
- nanochat introduces a "Complexity Dial" (`--depth` parameter) that auto-configures all hyperparameters

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
