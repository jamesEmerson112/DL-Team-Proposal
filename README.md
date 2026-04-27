# DL Team Proposal

This project investigates efficient language model training under extreme compression constraints through [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge. The objective is to train the best possible language model within a 16 MB artifact budget and a 10-minute wall-clock limit on 8×H100 GPUs, then analyze which architectural and optimization techniques contribute most to model quality as measured by validation bits-per-byte (BPB).

## Team

| Member | GitHub |
|--------|--------|
| James Vo | [@jamesEmerson112](https://github.com/jamesEmerson112) |
| — | [@Ashray14](https://github.com/Ashray14) |
| — | [@ialeksic3](https://github.com/ialeksic3) |
| — | [@sranganath2](https://github.com/sranganath2) |

## Current Status

| Metric | Value |
|--------|-------|
| Best val_bpb | **1.2077** (Run 11, SP8192 combo slim + TTT, 8×H100) |
| PG Baseline | 1.2244 |
| Leaderboard SOTA | 1.0810 |
| GPU setups | 2×H100 (exploration) / 8×H100 (competition) |
| Runs completed | 30+ |
| Deadline | April 30, 2026 |

## Repository Structure

```
├── parameter-golf/          <- forked competition repo (submodule)
├── runs/                    <- experiment launch scripts
│   └── configs/             <- per-experiment .env configs (21 configs)
├── tools/                   <- plotting utilities (learning curves)
├── docs/
│   ├── parameter-golf/      <- competition overview, findings, paper survey
│   ├── James_test/          <- research notes (numbered 00-20, compute-plan)
│   ├── runs/                <- early experiment logs (Apr 19)
│   ├── official/            <- course deliverables (proposal, final paper)
│   ├── meetings/            <- meeting notes
│   └── plots/               <- generated charts
```

## Getting Started

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/jamesEmerson112/DL-Team-Proposal.git
cd DL-Team-Proposal
```

### 2. Install dependencies

Ensure you are running the latest version of PyTorch (2.6+) before installing project dependencies:

```bash
pip install --upgrade torch
pip install -r parameter-golf/requirements.txt
```

### 3. Choose an experiment config

Configs are `.env` files in `runs/configs/`, grouped by purpose:

**Exploration**

| Config | GPUs | Purpose |
|--------|------|---------|
| `smoke_test.env` | 1 | Quick sanity check (reduced data) |
| `explore_1gpu.env` | 1 | Single-GPU exploration, 10 min |
| `explore_2gpu.env` | 2 | Multi-GPU exploration, 10 min |
| `competition_8gpu.env` | 8 | Full competition submission |

**Technique Ablations (SP1024)**

| Config | Technique |
|--------|-----------|
| `gated_attn_elementwise.env` | Elementwise gated attention |
| `gated_attn_headwise.env` | Headwise gated attention |
| `headwise_qkgain5.env` | Headwise + QK-Gain 5.0 |
| `leaky_relu2.env` | LeakyReLU² activation |
| `leaky_relu2_headwise.env` | LeakyReLU² + headwise gated attention |

**Competition (SP8192)**

| Config | Technique |
|--------|-----------|
| `sp8192_combo.env` | SP8192 + TTT + LeakyReLU² + QKG5 + headwise (full) |
| `sp8192_combo_slim.env` | Same but MODEL_DIM=448 to fit 16 MB budget |
| `sp8192_combo_slim_nottt.env` | Slim combo without TTT (ablation) |

**SLM Experiments (Selective Language Modeling)**

| Config | SLM Ratio |
|--------|-----------|
| `slm_test.env` | k=0.6 (smoke test) |
| `slm_sweep_40.env` ... `slm_sweep_90.env` | k=0.4, 0.5, 0.7, 0.8, 0.9 |
| `leaky_relu2_slm.env` | LeakyReLU² + SLM k=0.6 |
| `headwise_qkgain5_slm.env` | Headwise + QKG5 + SLM k=0.6 |
| `leaky_relu2_headwise_slm.env` | LeakyReLU² + headwise + SLM k=0.6 |

### 4. Run

```bash
source runs/configs/<config>.env
bash runs/parameter_golf_baseline.sh
```

The final `val_bpb` value in the output is the metric to compare against the baseline (1.2244).

### Run Scripts

| Script | Purpose |
|--------|---------|
| `parameter_golf_baseline.sh` | Main PG training pipeline — downloads data, trains model, checks 16 MB artifact budget |
| `run_all_2gpu.sh` | Sequential runner for all 2×H100 experiments (14 runs) |
| `run_slm_validation_2gpu.sh` | SLM validation — 4 focused runs with preflight check |
| `parameter_golf_8gpu_3seed_run.sh` | 3-seed reproducibility run for submission |

```bash
# PG training with a config
source runs/configs/explore_2gpu.env && bash runs/parameter_golf_baseline.sh

# Override a single variable inline
NUM_KV_HEADS=1 bash runs/parameter_golf_baseline.sh

# Run all 2×H100 experiments
bash runs/run_all_2gpu.sh

# SLM validation (verifies SLM code is present before running)
bash runs/run_slm_validation_2gpu.sh
```

### Config Environment Variables

Configs in `runs/configs/` are `.env` files that export the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time limit (seconds) |
| `RUN_ID` | `baseline_sp1024` | Run name for logs |
| `VAL_LOSS_EVERY` | `1000` | Evaluate validation BPB every N steps |
| `TRAIN_LOG_EVERY` | `100` | Log training loss every N steps |
| `GATED_ATTN` | `none` | Attention variant (`none`, `elementwise`, `headwise`) |
| `ACTIVATION` | `relu2` | MLP activation (`relu2`, `leaky_relu2`) |
| `VOCAB_SIZE` | `1024` | Tokenizer vocab size (`1024`, `8192`) |
| `MODEL_DIM` | `512` | Model embedding dimension (`512`, `448`) |
| `NUM_KV_HEADS` | `4` | KV head count (1 = MQA, 4 = GQA) |
| `QK_GAIN_INIT` | — | QK attention gain scaling (e.g., `5.0`) |
| `TTT_MODE` | `none` | Test-time training (`none`, `score_first`) |
| `SLM_ENABLED` | `0` | Selective Language Modeling (`0`, `1`) |
| `SLM_RATIO` | `0.6` | Fraction of tokens to keep (0.4-0.9) |
| `DATA_PATH` | SP1024 default | Dataset directory |
| `TOKENIZER_PATH` | SP1024 default | Tokenizer model file |

## Experiment Results

Top runs sorted by BPB (best first). Full results in [`docs/parameter-golf/findings.md`](docs/parameter-golf/findings.md).

### 8×H100 Runs

| Run | Technique | val_bpb | Budget? |
|-----|-----------|---------|---------|
| **11** | **SP8192 combo slim + TTT** | **1.2077** | **Yes (15.35 MB)** |
| 10 | SP8192 combo + TTT | 1.1872 | No (19.41 MB) |

3-seed reproducibility: mean 1.2073, std ±0.0006 BPB.

### 2×H100 Runs (Exploration)

| Run | Technique | val_bpb | Budget? |
|-----|-----------|---------|---------|
| A | SP8192 combo slim + TTT | 1.2411 | Yes |
| H | SP8192 combo slim (no TTT) | 1.2432 | Yes |
| 3 | Elementwise gated attn | 1.2602 | No (17.87 MB) |
| 7 | LeakyReLU² | 1.2641 | Yes |
| 6v2 | GQA baseline | 1.2649 | Yes |
| 2 | Headwise gated attn | 1.2653 | Yes |
| 9 | Headwise + QK-Gain 5.0 | 1.2719 | Yes |
| 4 | MQA (1 KV head) | 1.2761 | No (16.84 MB) |

## Research Notes

### Architecture and Leaderboard Analysis

- [PG vs nanochat Architecture](docs/James_test/13_pg-vs-nanochat-architecture.md) — structural comparison of the two codebases
- [modded-nanogpt Lineage](docs/James_test/14_modded-nanogpt-lineage.md) — genealogy: nanoGPT -> modded-nanogpt -> PG / nanochat
- [Paper Summaries for Team](docs/James_test/15_paper-summaries-for-team.md) — quick reference for key papers
- [PG Leaderboard Techniques](docs/James_test/16_pg-leaderboard-techniques.md) — deep dive into the top 8 techniques
- [PG Leaderboard Annotated](docs/James_test/17_pg-leaderboard-annotated.md) — all 30+ leaderboard entries decoded
- [LLM Parameter Anatomy](docs/James_test/18_llm-parameter-anatomy.md) — understanding parameter counts and model sizing
- [Gated Attention & SP8192](docs/James_test/19_gated-attention-and-sp8192.md) — two key techniques that shaped our approach
- [Rho-1 / Selective Language Modeling](docs/James_test/20_rho1-selective-language-modeling.md) — SLM implementation and testing plan
- [Compute Plan](docs/James_test/compute-plan.md) — GPU access, cost estimates, cloud alternatives

### Parameter Golf

- [Overview & Setup](docs/parameter-golf/00_overview.md)
- [Findings & Insights](docs/parameter-golf/findings.md) — full experiment results, technique analysis, leaderboard context
- [NeurIPS Paper Survey](docs/parameter-golf/neurlps-paper-survey.md) — 29 papers with actionable techniques ranked by impact and effort
- [Submission Guide](docs/parameter-golf/submission-guide.md)

## Tools

**Learning Curve Plotter** — parse Parameter Golf training logs and produce publication-ready charts.

```bash
# Single run (two subplots: train_loss + val_bpb)
python tools/plot_curves.py logs/run1.txt --mode single

# Compare multiple runs (overlay val_bpb curves)
python tools/plot_curves.py logs/run1.txt logs/run2.txt --name "GQA" "MQA" --mode compare

# Export to CSV for Overleaf
python tools/plot_curves.py logs/*.txt --mode csv
```

Plots are saved to `docs/plots/`. Dark theme, PG Baseline (1.2244) and SOTA (1.0810) reference lines included.

## Course Deliverables

This project is part of CS 7643 Deep Learning. Course requirements and templates are in [`docs/official/`](docs/official/):

- [Proposal Requirements](docs/official/requirements/proposal-requirements.md)
- [Final Paper Requirements](docs/official/requirements/final-paper.md)
- [Training Plan](docs/official/training-plan.md)

## License

[MIT](LICENSE)
