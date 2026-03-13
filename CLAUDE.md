# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Deep Learning Team Proposal** repository focused on evaluating and comparing minimal LLM training frameworks: **nanoGPT**, **llm.c**, and **nanochat** (all by Andrej Karpathy). The goal is to produce a team proposal document recommending an approach for a DL project.

## Repository Structure

- `James_paper/` — Team member contributions (research notes, drafts)
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
- The team has an existing comparison table covering language, dependencies, performance, parameters, training stages, cost, and learning curve across all three frameworks
- llm.c offers ~7% faster performance than PyTorch with minimal dependencies (pure C/CUDA)
- nanochat introduces a "Complexity Dial" (`--depth` parameter) that auto-configures all hyperparameters
