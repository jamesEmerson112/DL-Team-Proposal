# Compute Plan — GPU Access & Cost Estimates

> **Status:** Draft — needs team discussion
> **Problem:** PACE (Georgia Tech HPC) queue is 24+ hours with no guarantee of GPU allocation.
> **Decision:** Focus all experiments on the 16 MB Parameter Golf model, not the full 560M nanochat model.

---

## Why 16 MB Only

| | Full nanochat (560M) | Parameter Golf (16 MB) |
|---|---|---|
| Time per run | ~4 hrs on 8xH100 | ~10 min on 8xH100 |
| Cost per run | ~$100 | ~$4 |
| 20 experiments | ~$2,000 | ~$80 |
| Feasibility | Not feasible | Feasible |

The learning curve comparisons (our core deliverable) work at either scale. The 16 MB model is small enough to iterate quickly and cheaply.

---

## Single-GPU Estimates (16 MB Model)

The 16 MB model (9 layers, 512 hidden dims, 1024 vocab) fits on any modern GPU. nanochat supports single-GPU training via gradient accumulation (8x slower than 8-GPU).

| Setup | Est. Time/Run | Cost/Run | Best For |
|-------|--------------|----------|----------|
| 8xH100 | 10 min | ~$4 | Final validation, competition submission |
| 1xH100 | ~80 min | ~$4 | Real experiments (RunPod/Lambda) |
| 1xA100 (40GB) | ~90-120 min | ~$2-3 | Real experiments (Colab Pro, Vast.ai) |
| 1xT4 (Colab free) | ~4-6 hrs | $0 | Development, debugging, smoke tests |
| Apple Silicon (local) | varies | $0 | Code changes, quick sanity checks |

---

## Recommended Strategy

1. **Develop & debug** on Colab free (T4) or local Apple Silicon — $0
2. **Run experiments** on single A100 (Colab Pro ~$10/mo, or Vast.ai ~$1-2/hr) — ~$80 for 20 runs
3. **Final runs** on 8xH100 via Parameter Golf RunPod credits (free) — to match competition environment

**Total estimated cost:** ~$80-100 split across 3 team members

---

## GPU Cloud Alternatives (PACE Queue Workaround)

PACE queues are 24+ hours with no GPU guarantee. Here are alternatives, ordered by cost.

### Free / Near-Free Options (Try These First)

| Option | What You Get | How to Access |
|--------|-------------|---------------|
| **Parameter Golf RunPod Credits** | Free GPU credits from the $1M pool | Apply at [modelcraft.runpod.io](https://modelcraft.runpod.io) — we're eligible as Parameter Golf participants |
| **GT Departmental Clusters** | Bypass PACE entirely | SysML cluster (8xA40 nodes), Skynet (~568 GPUs) — ask your advisor or lab for access |
| **Google Cloud Education** | $300 free credits | Sign up with .edu email at [cloud.google.com/edu](https://cloud.google.com/edu) |
| **DigitalOcean** | $200 new account credit | [digitalocean.com](https://www.digitalocean.com) — GPU droplets available |
| **Azure for Students** | $100/year | [azure.microsoft.com/free/students](https://azure.microsoft.com/en-us/free/students/) — no credit card needed |
| **Kaggle Notebooks** | 30 hrs/week free P100 | [kaggle.com](https://www.kaggle.com) — good for prototyping, not ideal for long runs |

### Cheapest Paid Options (Single GPU, On-Demand)

| Provider | GPU | $/hr | Notes |
|----------|-----|------|-------|
| **Lambda** | A100 80GB | $1.10 | [lambdalabs.com](https://lambdalabs.com) — often sold out, check availability |
| **RunPod** | A100 SXM | $1.39 | [runpod.io](https://www.runpod.io) — spot instances even cheaper |
| **Vast.ai** | H100 | ~$1.87 | [vast.ai](https://vast.ai) — variable pricing, community marketplace |
| **RunPod** | H100 SXM | $2.69 | [runpod.io](https://www.runpod.io) — guaranteed availability |
| **Jarvis Labs** | H100 | $2.99 | [jarvislabs.ai](https://jarvislabs.ai) — pre-built ML images |

### Cost Estimate for Our 20 Experiments

- Each run: ~80-120 min on a single A100/H100
- Total GPU-hours needed: ~33 hrs (20 runs × ~100 min avg)

| Scenario | Provider | Total Cost |
|----------|----------|-----------|
| Best case | Parameter Golf credits (free) | **$0** |
| Budget | Lambda A100 @ $1.10/hr | **~$36** |
| Mid-range | RunPod A100 @ $1.39/hr | **~$46** |
| Comfortable | RunPod H100 @ $2.69/hr | **~$89** |
| Upper bound | Jarvis Labs H100 @ $2.99/hr | **~$99** |

**Recommendation:** Apply for Parameter Golf RunPod credits first (free). If unavailable, Lambda or RunPod A100 instances keep the whole project under $50.

---

## Action Items

- [ ] Apply for Parameter Golf RunPod credits at modelcraft.runpod.io
- [ ] Check access to GT departmental clusters (SysML, Skynet) — ask advisor
- [ ] Claim Google Cloud Education $300 credits with .edu email
- [ ] Set up Lambda or RunPod account as paid fallback
- [ ] Verify nanochat single-GPU training works with Parameter Golf config
- [ ] Run baseline on single GPU, confirm learning curves match expected behavior

---

## Impact on Proposal

None. The approach draft already frames everything around Parameter Golf constraints. We're just dropping the full 560M pipeline, which was never the core contribution — the learning curve comparison across techniques is.
