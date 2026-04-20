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
