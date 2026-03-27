# Running This Project on Cloud GPUs

No DGX Spark? You can run the entire curriculum on rented cloud GPUs. This guide covers the easiest options.

## Requirements

- **Minimum GPU**: 1x 80GB VRAM (A100 80GB or H100)
- **Why 80GB**: GPT-OSS 120B weighs ~65GB in mxfp4, plus KV cache and overhead
- **Disk**: 90GB+ persistent storage for model weights
- **QLoRA training**: Also works on any 80GB GPU

## Provider Comparison

| Provider | Best GPU for Price | $/hr | Setup | Best For |
|----------|-------------------|------|-------|----------|
| **RunPod** | A100 80GB PCIe | $1.19-1.39 | Easiest — vLLM Docker template, per-second billing | First-timers, quick experiments |
| **Lambda Labs** | H100 80GB SXM | $2.49-2.99 | Easy — SSH-ready VMs, PyTorch pre-installed | Reliability, longer sessions |
| **Vast.ai** | H100 80GB | $1.65+ | Moderate — marketplace pricing, variable reliability | Budget-conscious, spot workloads |

Prices as of March 2026. Check provider sites for current rates.

## Quick Start: RunPod (Recommended)

RunPod has built-in vLLM support and is the easiest path to running this project.

### 1. Launch a Pod

- Go to [runpod.io](https://www.runpod.io), create an account
- Deploy a GPU Pod with **A100 80GB** (or H100 for faster inference)
- Select the **RunPod PyTorch** template from the dropdown, or enter image manually: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- Set volume disk to **90GB+**, mounted at `/workspace`

### 2. Install vLLM

```bash
# SSH into your pod, then:
pip install vllm>=0.18.0 openai requests
```

### 3. Serve GPT-OSS 120B

```bash
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --download-dir /workspace/hf-cache
```

Model download takes ~15-30 min on first run. Subsequent starts load from cache.

### 4. Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or use the project's example scripts:
```bash
git clone https://github.com/mrbam/dgx-spark-ai.git
cd dgx-spark-ai
python examples/chat_completion.py
```

## What Works As-Is on Cloud

| Component | Status | Notes |
|-----------|--------|-------|
| `training/fine_tune.py` | Works | QLoRA training runs on any 80GB GPU |
| `training/fine_tune_peft.py` | Works | PEFT/TRL fallback, platform-independent |
| `training/merge_lora.py` | Works | Merge LoRA adapters |
| `examples/` | Works | All OpenAI SDK examples |
| `inference/test_inference.py` | Works | API smoke tests |

## What to Skip or Adapt

| Component | Action | Why |
|-----------|--------|-----|
| `systemd/` | Skip | Cloud pods don't use systemd; just run vLLM directly |
| `cline/` | Skip | Unless you tunnel back to your local machine (e.g., Tailscale) |
| `inference/start_vllm_gptoss.sh` | Adapt | Has DGX Spark-specific library paths; use the vLLM command above instead |
| `Makefile` targets (`make serve`, etc.) | Adapt | Hardcoded for local Spark; run vLLM commands directly |

## Cost Estimates

| Task | Duration | Cost (A100) | Cost (H100) |
|------|----------|-------------|-------------|
| Quick test (serve + chat) | ~2 hrs | $2-3 | $5-6 |
| Full curriculum walkthrough | ~8 hrs | $10-12 | $20-24 |
| Fine-tuning run (QLoRA) | ~1 hr | $1-2 | $2-3 |
| Download model (first time) | ~30 min | $0.60-0.70 | $1.25-1.50 |

**Tip**: Use persistent volumes so you don't re-download the model each time. Stop your pod when not in use — billing is per-second on RunPod.

## Reducing Costs

- **Use A100 80GB PCIe** instead of H100 — half the price, still runs GPT-OSS 120B fine
- **Reduce context length** if you don't need 32K tokens: `--max-model-len 16384` or `8192`
- **Use Community Cloud** on RunPod for lower rates (less availability guarantees)
- **Train on smaller models first** (7B/32B) before committing GPU hours to 120B
