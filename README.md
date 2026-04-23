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
| Best val_bpb | **1.2735** (2×H100, MQA) |
| PG Baseline | 1.2244 |
| Leaderboard SOTA | 1.0810 |
| Current GPU setup | 2×H100 |
| Runs completed | 4 |
| Deadline | April 30, 2026 |

## Repository Structure

```
├── parameter-golf/          ← forked competition repo (submodule)
├── nanochat/                ← Karpathy's training pipeline (submodule)
├── runs/                    ← experiment launch scripts
│   └── configs/             ← per-experiment .env configs
├── tools/                   ← plotting utilities (learning curves)
├── docs/
│   ├── parameter-golf/      ← competition overview, experiments, findings
│   ├── James_notes/         ← research notes (architecture, leaderboard, papers)
│   ├── runs/                ← per-run experiment logs
│   ├── official/            ← course deliverables (proposal, final paper)
│   ├── meetings/            ← meeting notes
│   └── plots/               ← generated charts
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

Available configurations in `runs/configs/`:

| Config | GPUs | Wall Clock | Purpose |
|--------|------|------------|---------|
| `smoke_test.env` | 1 | 5 min | Quick sanity check (tiny dataset) |
| `explore_1gpu.env` | 1 | 10 min | Budget single-GPU exploration |
| `explore_2gpu.env` | 2 | 10 min | Budget multi-GPU exploration |
| `competition_8gpu.env` | 8 | 10 min | Full competition submission |
| `gated_attn_elementwise.env` | 1 | 10 min | Gated attention ablation (elementwise) |
| `gated_attn_headwise.env` | 1 | 10 min | Gated attention ablation (headwise) |

### 4. Run

```bash
source runs/configs/<config>.env
bash runs/parameter_golf_baseline.sh
```

The final `val_bpb` value in the output is the metric to compare against the baseline (1.2244).

### Run Scripts

Three scripts are available in `runs/`:

| Script | Purpose |
|--------|---------|
| `parameter_golf_baseline.sh` | Main PG training pipeline — downloads data, trains model, checks 16 MB artifact budget |
| `nanochat_vs_pgolf.sh` | Apples-to-apples BPB comparison between PG and nanochat on the same dataset |
| `nanochat_single_gpu_d1.sh` | Train a minimal nanochat model (depth=1) on 1 GPU — sets up venv, downloads data, trains |

```bash
# PG training with a config
source runs/configs/explore_2gpu.env && bash runs/parameter_golf_baseline.sh

# Override a single variable inline
NUM_KV_HEADS=1 bash runs/parameter_golf_baseline.sh

# Compare PG vs nanochat engines
bash runs/nanochat_vs_pgolf.sh

# Nanochat smoke test (no config needed)
bash runs/nanochat_single_gpu_d1.sh
```

### Config Environment Variables

Configs in `runs/configs/` are `.env` files that export the following variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NGPUS` | `1` | Number of GPUs |
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time limit (seconds) |
| `RUN_ID` | `baseline_sp1024` | Run name for logs |
| `TRAIN_SHARDS` | all (80) | Limit data shards for fast testing |
| `VAL_LOSS_EVERY` | `1000` | Evaluate validation BPB every N steps |
| `TRAIN_LOG_EVERY` | `100` | Log training loss every N steps |
| `GATED_ATTN` | off | Attention variant (`elementwise` or `headwise`) |
| `NUM_KV_HEADS` | `4` | KV head count (1 = MQA, 4 = GQA) |

## Experiment Results

| Run | Date | GPUs | Change | Steps | val_bpb |
|-----|------|------|--------|-------|---------|
| GQA baseline | 2026-04-19 | 2×H100 | — | 1,770 | 1.3065 |
| MQA | 2026-04-19 | 2×H100 | `NUM_KV_HEADS=1` | 3,767 | **1.2735** |

Full experiment logs with hyperparameters and analysis are in [`docs/runs/`](docs/runs/).

## Research Notes

### Architecture and Leaderboard Analysis

- [PG vs nanochat Architecture](docs/James_notes/13_pg-vs-nanochat-architecture.md) — structural comparison of the two codebases
- [modded-nanogpt Lineage](docs/James_notes/14_modded-nanogpt-lineage.md) — genealogy: nanoGPT → modded-nanogpt → PG / nanochat
- [Paper Summaries for Team](docs/James_notes/15_paper-summaries-for-team.md) — quick reference for key papers
- [PG Leaderboard Techniques](docs/James_notes/16_pg-leaderboard-techniques.md) — deep dive into the top 8 techniques on the leaderboard
- [PG Leaderboard Annotated](docs/James_notes/17_pg-leaderboard-annotated.md) — all 30+ leaderboard entries decoded
- [Compute Plan](docs/James_notes/compute-plan.md) — GPU access, cost estimates, cloud alternatives

### Parameter Golf

- [Overview & Setup](docs/parameter-golf/00_overview.md)
- [Experiments Tracker](docs/parameter-golf/experiments.md)
- [Findings & Insights](docs/parameter-golf/findings.md)
- [NeurIPS Paper Survey](docs/parameter-golf/neurips-paper-survey.md) — 18 papers with actionable techniques ranked by impact and effort

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
