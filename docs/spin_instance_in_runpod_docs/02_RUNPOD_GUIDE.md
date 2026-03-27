# Running the GPT-OSS 120B Curriculum on RunPod

**The full DGX Spark curriculum — adapted for cloud GPUs**

No DGX Spark? No problem. This guide walks you through the same ~2.5 hour curriculum using a rented GPU on RunPod. By the end, you'll have served a 120-billion parameter model, chatted with it via API, built Python apps, and connected it to Cline as your AI coding assistant.

**Estimated cost**: ~$3-5 on an A100 80GB (~$1.19-1.39/hr)

**Prerequisites**:
- RunPod account ([runpod.io](https://www.runpod.io))
- Credit card for GPU rental
- Basic terminal comfort (cd, ls, curl, SSH)
- 2-3 hours of focused time

**Navigation**: [Module 1: RunPod Setup](#module-1-setting-up-your-runpod-pod-15-min) | [Module 2: Installing vLLM](#module-2-installing-vllm-on-runpod-20-min) | [Module 3: Starting the Model](#module-3-starting-your-first-model-45-min) | [Module 4: Your First Conversation](#module-4-your-first-conversation-30-min) | [Module 5: Building with Python](#module-5-building-with-python-30-min) | [Module 6: Cline Integration](#module-6-cline-integration-via-runpod-proxy-45-min) | [Module 7: Going Further](#module-7-going-further)

---

## Module 1: Setting Up Your RunPod Pod (15 min)

**Learning Objectives**: Create a cloud GPU environment capable of running GPT-OSS 120B

### 1.1 Why RunPod?

RunPod offers on-demand GPU rentals with:
- Per-second billing (no wasted money)
- Persistent volumes (model survives pod restarts)
- Built-in proxy URLs (access your server from anywhere)
- vLLM-friendly templates

**Cost context**: An A100 80GB costs ~$1.19-1.39/hr. The full curriculum takes ~2.5 hours, so budget ~$3-5 total.

### 1.2 Create Your Pod

1. **Sign up** at [runpod.io](https://www.runpod.io) and add credits ($10 is plenty)

2. **Deploy a GPU Pod**:
   - Click **Deploy** or **GPU Pods** → **Deploy**
   - **GPU**: Select **A100 80GB PCIe** (~$1.19/hr) or **A100 80GB SXM** (~$1.39/hr)
     - H100 works too (~$1.99/hr) but costs more for similar results here
   - **Template**: Select **RunPod PyTorch** from the template dropdown, or enter the image manually: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
   - **Volume Disk**: Set to **100GB**, mounted at `/workspace`
     - This is persistent — your model cache survives pod stops/restarts
   - **Container Disk**: Default (20GB) is fine
   - Click **Deploy**

> **Troubleshooting**: If you see `manifest for runpod/pytorch:latest not found: manifest unknown`, RunPod doesn't publish a `:latest` tag. Make sure you're using the full version tag (e.g., `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`) or selecting the PyTorch template from the dropdown instead of typing the image name manually.

3. **Wait for the pod to start** (1-2 minutes)

4. **Set up SSH access** (do this before your first connection):

   **Generate an SSH key** (skip if you already have one at `~/.ssh/id_ed25519`):
   ```bash
   ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_ed25519 -N ""
   ```

   **Add your public key to RunPod**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   Copy the output → RunPod dashboard → **Settings** → **SSH Public Keys** → paste and save.

   > **Important**: If you add the key after the pod is already running, you must **restart the pod** (stop → start) for it to pick up the new key. The pod ID and connection details may change after restart — check the Connect dialog again.

5. **Connect to your pod** — three options (easiest first):

   **Option A: Web Terminal** (no setup needed)
   - In your pod's Connect dialog, click **"Open web terminal"**
   - Works immediately, runs in your browser

   **Option B: SSH via RunPod proxy** (recommended for terminal use)
   ```bash
   ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
   ```
   Find the full command in your pod's Connect dialog under "SSH".

   **Option C: VS Code Remote SSH** (best for development)
   - Install the **Remote - SSH** extension in VS Code
   - Command Palette → **Remote-SSH: Connect to Host**
   - Enter: `<pod-id>@ssh.runpod.io`
   - Full IDE experience on the pod — file explorer, terminal, extensions

   **Option D: RunPod CLI (`runpodctl`)** (manage pods from your local terminal)

   Install on macOS:
   ```bash
   brew install runpod/runpodctl/runpodctl
   ```

   Or manually (Apple Silicon):
   ```bash
   wget https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-darwin-arm64 -O runpodctl
   chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/runpodctl
   ```

   Configure with your API key (find it at RunPod dashboard → Settings → API Keys):
   ```bash
   runpodctl config --apiKey YOUR_API_KEY
   ```

   Then manage pods directly from your local terminal:
   ```bash
   runpodctl get pod                          # List your pods
   runpodctl ssh <pod-id>                     # SSH into a pod
   runpodctl send <file> <pod-id>:/workspace/ # Transfer files to pod
   runpodctl receive <pod-id>:/workspace/file # Download files from pod
   runpodctl stop pod <pod-id>                # Stop pod (saves money)
   runpodctl start pod <pod-id>               # Restart pod
   ```

   This lets you create, start, stop, and SSH into pods without ever opening the RunPod dashboard.

### 1.3 Verify Your GPU

```bash
nvidia-smi
```

You should see something like:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx       Driver Version: 550.xx       CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A100-SXM4-80GB On | 0000:00:04.0  Off |                    0 |
+-------------------------------+----------------------+----------------------+
```

Key things to confirm:
- **80GB** memory (not 40GB)
- **CUDA** version 12.x+

```bash
# Check total GPU memory
nvidia-smi --query-gpu=memory.total --format=csv
# Should show: 81920 MiB (80GB)
```

**Cost tip**: The billing clock is running! When you're taking a break or reading ahead, **stop your pod** from the RunPod dashboard (your volume persists). Restart it when you're ready to continue.

### 1.4 Understanding the Difference: Cloud vs. DGX Spark

| | DGX Spark | RunPod A100 80GB |
|---|-----------|-----------------|
| **Memory** | 128GB unified CPU+GPU | 80GB dedicated GPU VRAM |
| **Architecture** | Grace-Blackwell (GB10) | Ampere (A100) |
| **CUDA** | 13.0 | 12.x |
| **Cost** | ~$3,000 one-time | ~$1.19-1.39/hr |
| **mxfp4 hardware accel** | Yes (native) | No (software fallback) |
| **Always available** | Yes (on your desk) | No (need internet + credits) |

GPT-OSS 120B fits on both: ~65GB in mxfp4 quantization, with enough headroom on the A100's 80GB for KV cache.

<details>
<summary><b>Deep Dive: Why 80GB Is Enough</b></summary>

GPT-OSS 120B uses **mxfp4 microscaling quantization**:

- **Full precision (FP16)**: 120B params × 2 bytes = ~240GB — won't fit
- **mxfp4 quantized**: 120B params × ~0.5 bytes = ~65GB — fits on 80GB!
- **Remaining ~15GB**: Used for KV cache (attention memory), CUDA workspace, and overhead

The A100 doesn't have native mxfp4 hardware acceleration like the GB10 (Blackwell), but vLLM handles the dequantization in software. You may see slightly lower tokens/sec compared to a Spark, but it works.

**Memory budget on A100 80GB**:
```
80 GB Total VRAM
├── Model weights (mxfp4):     ~65 GB
├── KV cache (32K context):    ~10 GB (with --gpu-memory-utilization 0.90)
├── CUDA workspace:            ~3 GB
└── Headroom:                  ~2 GB
```

This is tighter than the Spark's 128GB, which is why we use `--gpu-memory-utilization 0.90` to tell vLLM to use up to 90% of VRAM.

</details>

**Checkpoint**: You have a running pod with an A100 80GB (or H100) and can access a terminal.

---

## Module 2: Installing vLLM on RunPod (20 min)

**Learning Objectives**: Install vLLM, clone the project, and prepare to serve models

### 2.1 Install Dependencies

```bash
# Install vLLM and the OpenAI SDK
pip install vllm>=0.18.0 openai requests
```

This installs:
- **vLLM**: The inference engine
- **OpenAI SDK**: For testing via Python
- **requests**: HTTP library for examples

```bash
# Verify installation
vllm --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

You should see:
```
vLLM 0.18.0 (or newer)
PyTorch 2.x.x+cu12x, CUDA: True
```

### 2.2 Clone the Project

```bash
cd /workspace
git clone https://github.com/mrbam/dgx-spark-ai.git
cd dgx-spark-ai
```

**What you're looking at**:

```
dgx-spark-ai/
├── inference/          # Model serving scripts (vLLM)
│   ├── start_vllm_gptoss.sh    # Start GPT-OSS 120B (Spark-specific)
│   ├── stop_vllm.sh            # Stop vLLM
│   ├── health_check.sh         # Full pipeline health check
│   └── test_inference.py       # API smoke tests
├── training/           # Fine-tuning pipelines
│   ├── fine_tune.py            # QLoRA via Unsloth (fast)
│   ├── fine_tune_peft.py       # Standard PEFT/TRL
│   └── merge_lora.py           # Merge adapters
├── cline/             # Cline CLI integration
│   └── setup_cline_local.sh    # Auto-configure Cline
├── examples/          # Usage examples  <-- These all work on RunPod!
│   ├── chat_completion.py      # Basic chat
│   ├── streaming_example.py    # Real-time streaming
│   └── batch_inference.py      # Concurrent requests
├── docs/              # Deep-dive documentation
├── Makefile           # Convenience commands (Spark-specific)
└── README.md          # Original Spark curriculum
```

**Note**: On RunPod, we'll run vLLM commands directly instead of using the Makefile (which has Spark-specific library paths). The `examples/` and `training/` directories work as-is.

### 2.3 Understanding GPT-OSS 120B

**What is GPT-OSS?**
- OpenAI's first fully open-weight model
- 120 billion parameters (117B actual, MoE architecture with 5.1B active per token)
- Released with **mxfp4 quantization** (microscaling FP4)
- Ships at ~65GB instead of ~240GB (FP16)
- Competitive with much larger models due to efficient architecture

**What is mxfp4 quantization?**
- **4-bit floating point** with microscaling
- Groups weights into blocks, each with its own scale factor
- Near-FP16 quality at 1/4 the size
- Hardware-accelerated on Blackwell (GB10), software-decoded on A100/H100
- This is why 120B parameters fit in 65GB

**Model location**: Models cache at `~/.cache/huggingface/hub/` by default. On RunPod, we'll use `/workspace/hf-cache` (persistent volume) so you don't re-download on pod restart.

### 2.4 Pre-download the Model

Download to persistent storage so it survives pod restarts:

```bash
# This downloads ~65GB (takes 15-30 minutes depending on connection)
huggingface-cli download openai/gpt-oss-120b --local-dir /workspace/hf-cache/openai/gpt-oss-120b
```

**Cost tip**: The download takes 15-30 min where you're paying for GPU time but not using the GPU. This is unavoidable on the first run. On subsequent pod starts, the model loads from your persistent volume in ~60 seconds.

**Checkpoint**: `vllm --version` returns 0.18.0+, the project is cloned, and the model is downloading (or downloaded).

<details>
<summary><b>Deep Dive: How vLLM Differs From Other Inference Engines</b></summary>

**vLLM vs. Ollama**:
- Ollama: Easy setup, single-user, GGUF models, llama.cpp backend
- vLLM: Production-grade, OpenAI API, multi-user, continuous batching

**vLLM vs. llama.cpp**:
- llama.cpp: CPU-friendly, GGUF format, great for smaller hardware
- vLLM: GPU-optimized, safetensors/HF format, better throughput

**vLLM vs. TGI (Text Generation Inference)**:
- TGI: HuggingFace's official server, good Docker support
- vLLM: Faster (PagedAttention), better batching, larger model support

**Why vLLM for this curriculum?**
1. **PagedAttention**: Manages GPU memory like an OS manages RAM (reduces waste by ~50%)
2. **Continuous batching**: Processes multiple requests simultaneously
3. **OpenAI compatibility**: Works with any OpenAI client
4. **Native mxfp4 support**: Handles GPT-OSS quantization
5. **Production-ready**: Used by major AI companies

</details>

**Challenge 1**: Check how much disk space the model uses after download:
```bash
du -sh /workspace/hf-cache/openai/gpt-oss-120b
```
Compare with the theoretical size (120B params × 4 bits / 8 bits per byte).

---

## Module 3: Starting Your First Model (45 min)

**Learning Objectives**: Start the vLLM server, understand the startup process, monitor resources, and troubleshoot common issues

### 3.1 Starting vLLM

Time to bring GPT-OSS 120B to life! Run this command:

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

**What happens next** (60-120 seconds):
1. **Model architecture detection**: Loads custom `GptOssForCausalLM` class
2. **Weight loading**: Reads safetensor files (~65GB) into GPU memory
3. **Engine initialization**: Prepares vLLM scheduler and workers
4. **Server start**: OpenAI-compatible API on port 8000

You'll see a **lot** of log output. Here's what to look for:

```
INFO: Loading model weights...
Loading safetensors checkpoint shards:   7%|█▎              | 1/15 [00:02<00:31,  2.26s/it]
Loading safetensors checkpoint shards:  20%|███▋            | 3/15 [00:06<00:24,  2.08s/it]
...
INFO: Started server process [12345]
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**When you see "Started server process"** — you're ready!

### 3.2 Understanding the Flags

Each flag serves a critical purpose:

| Flag | What It Does | Why |
|------|--------------|-----|
| `--host 0.0.0.0` | Listen on all interfaces | Required for RunPod proxy access |
| `--port 8000` | API port | Standard vLLM port |
| `--trust-remote-code` | Allow custom model code | GPT-OSS uses a custom architecture |
| `--enforce-eager` | Disable CUDA graphs | Saves ~10GB GPU memory (critical on 80GB) |
| `--max-model-len 32768` | Context window limit | 32K tokens (longer = more KV cache) |
| `--gpu-memory-utilization 0.90` | Use 90% of VRAM | Tighter fit on 80GB than 128GB Spark |
| `--download-dir` | Model cache location | Points to persistent volume |

**Why `--enforce-eager` matters**:
- CUDA graphs = pre-compiled execution plans (faster but memory-hungry)
- With graphs: ~75GB memory needed
- Without graphs (eager mode): ~65GB memory needed
- A100 has 80GB, so eager mode is essential to leave room for KV cache

**Why `--gpu-memory-utilization 0.90`**:
- On the Spark (128GB), the default 0.90 leaves ~13GB free — plenty
- On the A100 (80GB), 0.90 = 72GB for vLLM, leaving 8GB for CUDA overhead
- Don't go above 0.95 or you'll hit OOM errors

### 3.3 Monitoring the Model Load

Open a **second terminal** (RunPod web terminal supports tabs, or SSH in again):

```bash
watch -n 2 nvidia-smi
```

You'll see GPU memory climb from ~2GB → ~67GB over 60 seconds as weights load.

### 3.4 Accessing from Outside the Pod

RunPod provides a **proxy URL** for each exposed port. To find it:

1. Go to your RunPod dashboard
2. Click on your pod
3. Look at the **Ports** section
4. Find port **8000** — you'll see a URL like:
   ```
   https://<pod-id>-8000.proxy.runpod.net
   ```

You can test this from your **local machine**:
```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/models
```

This proxy URL is how you'll connect Cline later (Module 6).

### 3.5 Common Startup Issues

**Problem 1: CUDA Out of Memory**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Cause**: Model + KV cache exceed 80GB.

**Fixes**:
```bash
# Reduce context length (less KV cache)
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --enforce-eager \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --download-dir /workspace/hf-cache

# Or reduce memory utilization
    --gpu-memory-utilization 0.85
```

---

**Problem 2: Port Already in Use**
```
OSError: [Errno 98] Address already in use
```

**Fix**:
```bash
# Kill existing vLLM process
pkill -f "vllm serve"
sleep 3
# Restart
```

---

**Problem 3: Slow Startup (>3 minutes)**

**First time**: Compiling CUDA kernels is normal. Future runs are faster.

**Every time**: Check if the model is downloading instead of loading from cache:
```bash
ls /workspace/hf-cache/openai/gpt-oss-120b/
```
If empty, the model needs to download first (~15-30 min).

### 3.6 Verifying Success

In another terminal:

```bash
curl http://localhost:8000/v1/models
```

You should get:
```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-oss-120b",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 32768
    }
  ]
}
```

**Checkpoint**: Server shows "Started server process" and `curl http://localhost:8000/v1/models` returns JSON with the model ID.

<details>
<summary><b>Deep Dive: vLLM Architecture</b></summary>

**Inside vLLM**:

```
┌─────────────────────────────────────────────────┐
│  FastAPI Server (port 8000)                     │
│  └─ /v1/models, /v1/chat/completions, etc.      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  LLM Engine                                      │
│  ├─ Request queue                                │
│  ├─ Continuous batching scheduler                │
│  └─ KV cache manager (PagedAttention)           │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  Workers (GPU processes)                         │
│  ├─ Model loaded in GPU memory                   │
│  ├─ Tokenizer (Harmony for GPT-OSS)             │
│  └─ CUDA kernels (attention, sampling, etc.)    │
└──────────────────────────────────────────────────┘
```

**PagedAttention**:
- KV cache is divided into "pages" (like OS virtual memory)
- Pages can be swapped, shared between requests, etc.
- Reduces memory waste from padding by ~50%
- Critical for fitting large context windows

**Continuous Batching**:
- Traditional: Wait for all requests in batch to finish
- vLLM: Add/remove requests dynamically as they complete
- Result: Better GPU utilization, lower latency

**Why startup takes 60-120s**:
1. Load 65GB weights into memory: ~30-60s
2. Initialize CUDA contexts: ~10s
3. Compile Triton kernels (first time): ~20-40s
4. Build KV cache structures: ~5s
5. Start HTTP server: ~1s

</details>

**Challenge 2**: Time your startup! Measure how long from running the vLLM command to "Started server process". Compare with the 60-120s benchmark.

---

## Module 4: Your First Conversation (30 min)

**Learning Objectives**: Interact with the model via API, understand request/response formats, and explore GPT-OSS's unique reasoning capabilities

### 4.1 The OpenAI-Compatible API

vLLM exposes the same API as OpenAI's cloud service. This means:

- Any OpenAI client works out-of-the-box
- Familiar interface, cloud compute (your cloud this time!)
- Easy migration between local and cloud
- Swap `base_url` and you're done

**Endpoints available**:
- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — Chat with the model
- `POST /v1/completions` — Raw text completion
- `POST /v1/embeddings` — Generate embeddings (if model supports)

### 4.2 List Available Models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

**What this tells you**:
- **Model ID**: `openai/gpt-oss-120b` (use this in requests)
- **Max context**: 32,768 tokens
- **Owner**: `vllm` (your server, not OpenAI)

### 4.3 Your First Chat Message

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "Explain unified memory in one sentence."}
    ],
    "max_tokens": 100
  }' | python3 -m json.tool
```

**Response** (abbreviated):
```json
{
  "id": "chat-f7a8b9c0...",
  "object": "chat.completion",
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Unified memory allows the CPU and GPU to access the same physical memory pool without copying data between separate memory spaces.",
        "reasoning": "[Analysis] The user wants a concise explanation..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 31,
    "total_tokens": 49
  }
}
```

**Breaking it down**:

| Field | Meaning |
|-------|---------|
| `choices[0].message.content` | The actual answer |
| `choices[0].message.reasoning` | **GPT-OSS specialty**: Internal chain-of-thought |
| `usage.prompt_tokens` | Your input (18 tokens) |
| `usage.completion_tokens` | Model's output (31 tokens) |
| `finish_reason` | Why it stopped (`stop` = natural end, `length` = hit max_tokens) |

### 4.4 Understanding GPT-OSS's Reasoning Format

GPT-OSS is a **reasoning model** that generates structured thought channels:

**Three channels**:
1. **Analysis**: Understanding the request
2. **Commentary**: Meta-reasoning about the approach
3. **Final**: The actual response to the user

This is similar to OpenAI's o1/o3 models, but fully open and running on your rented GPU!

**Why this matters**:
- You can see *why* the model chose its response
- Helps debug unexpected outputs
- Enables advanced prompting techniques
- Great for educational use

### 4.5 Streaming Responses

For a better user experience, stream tokens as they're generated:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "stream": true
  }'
```

Watch tokens appear in real-time! This is the **Server-Sent Events (SSE)** protocol.

### 4.6 Experimenting with Parameters

```bash
# More creative (higher temperature)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Write a haiku about AI."}],
    "temperature": 0.9,
    "max_tokens": 50
  }' | python3 -m json.tool

# Deterministic (temperature 0)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0,
    "max_tokens": 10
  }' | python3 -m json.tool
```

**Key parameters**:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0 - 2.0 | Higher = more random/creative |
| `top_p` | 0.0 - 1.0 | Nucleus sampling (0.9 = top 90% probability mass) |
| `max_tokens` | 1 - 32768 | Maximum response length |
| `presence_penalty` | -2.0 - 2.0 | Discourage repetition |
| `frequency_penalty` | -2.0 - 2.0 | Penalize token frequency |

**Checkpoint**: You can send messages via curl and receive responses. Streaming works. You understand the response structure.

<details>
<summary><b>Deep Dive: Token Generation and Performance Metrics</b></summary>

**How token generation works**:

1. **Tokenization**: "Hello world" → `[15496, 1917]`
2. **Embedding**: Token IDs → vectors
3. **Transformer layers**: 120B parameters process vectors
4. **Next token prediction**: Probability distribution over vocab
5. **Sampling**: Pick next token based on temperature/top_p
6. **Repeat**: Until `</s>` or max_tokens

**Performance metrics**:

- **TTFT (Time To First Token)**: Latency until first output token
  - For GPT-OSS 120B on A100 80GB: ~500-2000ms
  - Depends on prompt length (longer = more prefill work)

- **Tokens/second**: Generation speed after first token
  - For GPT-OSS 120B on A100 80GB: ~5-15 tokens/sec
  - Varies with context length, batch size

- **Throughput vs. Latency**: Single user = optimize latency, Multiple users = optimize throughput

**Why local/cloud inference is "slow" compared to OpenAI API**:
- Cloud APIs use tensor parallelism (multiple GPUs)
- GPT-4/Claude use speculative decoding, optimized infrastructure
- 120B params on 1 GPU is impressive but compute-bound

**When self-hosted is worth it**:
- Zero per-token cost (just GPU rental)
- Complete privacy (data stays on your pod)
- No rate limits
- Customizable (fine-tuning, prompt templates)
- Learning and experimentation

</details>

**Challenge 3**: Ask the model about itself:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "What GPU am I running on? Be specific about the architecture."}],
    "max_tokens": 150
  }' | python3 -m json.tool
```
Does it know about the A100? Check the `reasoning` field to see its thought process!

---

## Module 5: Building with Python (30 min)

**Learning Objectives**: Use the OpenAI Python SDK, explore example patterns, and build your first application

### 5.1 Setting Up the OpenAI SDK

The OpenAI Python SDK works seamlessly with vLLM. Just point it at your server:

```python
from openai import OpenAI

# Create client pointing to vLLM on this pod
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-vllm"  # Can be anything (vLLM doesn't check)
)

# List models
models = client.models.list()
print(f"Available: {models.data[0].id}")

# Chat completion
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 5.2 Example 1: Basic Chat Completion

```bash
cd /workspace/dgx-spark-ai
python3 examples/chat_completion.py
```

**What it does**:
- Connects to local vLLM
- Sends a chat message
- Displays the response
- Shows token usage

**Key concepts**:
- **System message**: Sets the AI's behavior/personality
- **User message**: Your actual request
- **Assistant message**: The AI's response (in multi-turn conversations)

### 5.3 Example 2: Streaming Responses

```bash
python3 examples/streaming_example.py
```

Watch tokens appear one by one in your terminal. This is the same SSE protocol from Module 4, but handled by the Python SDK.

**Why streaming matters**:
- User sees output immediately (not waiting 10+ seconds)
- Better perceived performance
- Can cancel early if output goes off-track
- Essential for chat UIs

### 5.4 Example 3: Batch Processing

```bash
python3 examples/batch_inference.py
```

This sends 5 concurrent requests using Python's `ThreadPoolExecutor`.

**Why batch processing matters**:
- vLLM's **continuous batching** processes multiple requests simultaneously
- 5 concurrent requests ≠ 5x the time (more like 2-3x due to batching)
- Better GPU utilization
- Critical for multi-user scenarios

**Checkpoint**: All three examples run successfully. You understand basic chat, streaming, and batching patterns.

<details>
<summary><b>Deep Dive: vLLM Request Lifecycle</b></summary>

**What happens when you call `client.chat.completions.create()`:**

1. **Client side**: OpenAI SDK formats request as JSON, sends HTTP POST to `http://localhost:8000/v1/chat/completions`

2. **vLLM server receives**: FastAPI endpoint validates request, applies chat template (Harmony format for GPT-OSS), tokenizes input

3. **Request enters queue**: vLLM's scheduler adds request to queue, waits for available GPU slots

4. **Prefill phase** (first token): Entire prompt processed in parallel, generates KV cache, samples first output token. **This is the TTFT latency.**

5. **Decode phase** (subsequent tokens): One token generated per step, auto-regressive (each depends on previous), KV cache grows. **This is the tokens/sec throughput.**

6. **Batching magic**: Multiple requests in prefill/decode simultaneously, scheduler fills GPU compute capacity. **This is why concurrent requests are efficient.**

7. **Response sent**: If streaming: SSE chunks per token. If not: full response after completion.

</details>

**Challenge 4**: Modify `examples/chat_completion.py` to:
1. Ask the model to write a haiku about cloud computing
2. Save the response to a file called `haiku.txt`
3. Print the token usage

Hint: Use `open('haiku.txt', 'w')` and `f.write(content)`.

---

## Module 6: Cline Integration via RunPod Proxy (45 min)

**Learning Objectives**: Connect Cline CLI to your cloud-hosted model and experience AI-powered development

### 6.1 What is Cline?

[Cline](https://github.com/cline/cline) is an AI coding assistant that can:

- Read and write files in your project
- Execute terminal commands
- Browse the web for documentation
- Reason through complex multi-step tasks
- Use 44+ AI providers (including "OpenAI Compatible")

**Why Cline + Cloud vLLM is powerful**:
- Use a 120B parameter model as your coding assistant
- No per-token charges beyond GPU rental
- No rate limits
- See the model's reasoning traces

### 6.2 The RunPod Proxy Connection

Since Cline runs on your **local machine** and vLLM runs on your **RunPod pod**, you need a bridge. RunPod provides this automatically via proxy URLs.

**Find your proxy URL**:
1. Go to RunPod dashboard → your pod
2. Click the **Ports** section
3. Find port **8000**
4. Copy the URL: `https://<pod-id>-8000.proxy.runpod.net`

**Test it from your local machine**:
```bash
curl https://<pod-id>-8000.proxy.runpod.net/v1/models
```

You should get the same JSON response as `curl http://localhost:8000/v1/models` on the pod.

### 6.3 Installing Cline CLI (On Your Local Machine)

```bash
# Install via npm (on your local machine, not the pod)
npm install -g @cline/cli

# Verify installation
cline --version
```

### 6.4 Configure Cline to Use Your RunPod Model

```bash
# Replace <pod-id> with your actual pod ID from the proxy URL
cline auth \
    --provider openai \
    --apikey "local-vllm" \
    --modelid "openai/gpt-oss-120b" \
    --baseurl "https://<pod-id>-8000.proxy.runpod.net/v1"
```

**Verify configuration**:
```bash
cline config
```

Look for:
```
actModeApiProvider: openai
actModeOpenAiModelId: openai/gpt-oss-120b
openAiBaseUrl: https://<pod-id>-8000.proxy.runpod.net/v1
```

### 6.5 Your First Cline Task

```bash
cline "What model are you, and where are you running?"
```

Watch Cline:
1. Send request to your RunPod-hosted GPT-OSS 120B
2. Parse the response
3. Display it with formatting

**Note**: Responses may be slightly slower than local (network latency to RunPod + inference time), but perfectly usable.

### 6.6 Real Coding Tasks

#### Task 1: Generate a Python Function

```bash
cline "Write a Python function to calculate the Fibonacci sequence up to n terms. Include docstring and type hints."
```

#### Task 2: Work with Files

```bash
cline "Read the Makefile and explain what the 'serve' target does"
```

#### Task 3: Multi-Step Task

```bash
cline "Create a new Python file called 'hello.py' that prints 'Hello from GPT-OSS 120B on RunPod!' when run, then show me how to execute it."
```

### 6.7 Interactive Mode

Launch the full Cline TUI:

```bash
cline
```

Features:
- **Chat interface**: Conversational interaction
- **File tree**: See project structure
- **Command execution**: Run shell commands
- **Tool use**: Read files, write files, search, etc.

### 6.8 Plan vs. Act Modes

**Act Mode** (default): Executes tasks immediately. Best for simple, clear requests.

**Plan Mode**: Thinks through complex tasks first, creates a plan, gets your approval.

```bash
cline  # Enter interactive mode
# Type: /plan
# Then describe a complex task
```

### 6.9 Using Reasoning Traces

Because GPT-OSS 120B includes reasoning traces, you can see *why* Cline makes decisions:

```bash
cline config set showReasoning true
cline "What's the best way to optimize the vLLM startup script?"
```

You'll see the model's internal `[Analysis]`, `[Commentary]`, and `[Final]` thought channels.

**Checkpoint**: Cline successfully generates code using your RunPod-hosted GPT-OSS 120B model.

<details>
<summary><b>Deep Dive: How Cline Uses the OpenAI API</b></summary>

**Cline's architecture (with RunPod)**:

```
┌───────────────────────────────────────────────┐
│  Cline CLI (your local machine)               │
│  ├─ User interface (TUI or one-shot)          │
│  ├─ Conversation manager                      │
│  └─ Tool system (read_file, execute, etc.)    │
└────────────────┬──────────────────────────────┘
                 │
                 ▼ HTTPS (RunPod proxy)
┌────────────────────────────────────────────────┐
│  RunPod Pod                                    │
│  └─ vLLM Server (:8000)                       │
│     └─ /v1/chat/completions                   │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│  GPT-OSS 120B Model (on A100 80GB)            │
│  └─ Returns: reasoning + content               │
└────────────────────────────────────────────────┘
```

The only difference from the Spark setup is the network hop: `localhost` becomes `https://<pod-id>-8000.proxy.runpod.net`. Everything else — system prompts, tool calls, streaming — works identically.

</details>

**Challenge 5**: Ask Cline to create a Python script that queries the vLLM API at your RunPod URL, extracts the model name and max context length, and prints them formatted nicely. Then run it!

**Challenge 6** (Advanced): Use Cline in Plan Mode to add streaming support to the script from Challenge 5.

---

## Module 7: Going Further

You've completed the core curriculum!

### 7.1 Fine-Tuning Your Own Models

QLoRA fine-tuning works on any 80GB GPU. From the pod:

```bash
cd /workspace/dgx-spark-ai

# Install training dependencies
pip install -r training/requirements.txt

# Fine-tune with QLoRA (Unsloth for 2x speed)
python3 training/fine_tune.py --dataset your_data.jsonl

# Or use standard PEFT/TRL
python3 training/fine_tune_peft.py --dataset your_data.jsonl
```

**Dataset format**: JSONL with ChatML messages:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Deep dive**: See [Training Guide](05_TRAINING_GUIDE.md)

### 7.2 Switching Models

Try different models by stopping vLLM (Ctrl+C) and restarting with a different model:

```bash
# Fast 7B model for quick iterations
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --download-dir /workspace/hf-cache

# 32B coding specialist
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 16384 \
    --download-dir /workspace/hf-cache

# Back to GPT-OSS 120B
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --enforce-eager \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --download-dir /workspace/hf-cache
```

**Model comparison**:

| Model | Parameters | Memory | Context | Best For |
|-------|-----------|--------|---------|----------|
| **GPT-OSS 120B** | 120B (mxfp4) | ~65GB | 32K | Best quality, reasoning |
| Qwen2.5-7B-Instruct | 7B | ~15GB | 32K | Fast testing, general use |
| Qwen2.5-Coder-32B | 32B | ~24GB | 32K | Code generation |
| Llama-3.1-8B | 8B | ~17GB | 8K | General purpose |
| Llama-3.1-70B | 70B (quantized) | ~45GB | 8K | High quality |

### 7.3 Troubleshooting

**Quick diagnostics**:
```bash
# Check GPU
nvidia-smi

# Check vLLM processes
pgrep -fa "vllm serve"

# Check port
ss -tlnp | grep 8000

# Test API
curl http://localhost:8000/v1/models
```

**Most common issues**:

1. **CUDA OOM**: Reduce `--max-model-len` or `--gpu-memory-utilization`
2. **Port in use**: `pkill -f "vllm serve"` then restart
3. **Slow inference**: Expected for 120B on single GPU (~5-15 tok/s)
4. **Cline not connecting**: Verify your RunPod proxy URL includes `/v1` at the end

**Full troubleshooting guide**: [06_TROUBLESHOOTING.md](06_TROUBLESHOOTING.md)

---

## Cost Summary

| Task | Duration | Cost (A100 80GB) |
|------|----------|-----------------|
| Pod setup + model download | ~30 min | $0.60-0.70 |
| Modules 1-5 (API + Python) | ~2 hrs | $2.40-2.80 |
| Module 6 (Cline) | ~45 min | $0.90-1.05 |
| Fine-tuning (optional) | ~1 hr | $1.19-1.39 |
| **Full curriculum** | **~3.5 hrs** | **~$4-6** |

### Saving Money

- **Stop your pod** when taking breaks or reading ahead — billing is per-second
- **Use persistent volumes** — model cache survives pod stops, no re-download
- **Use A100 80GB PCIe** ($1.19/hr) over SXM ($1.39/hr) or H100 ($1.99/hr)
- **Reduce context** (`--max-model-len 16384`) if you don't need 32K tokens
- **Try smaller models first** (7B/32B) before committing time to 120B

---

## What You've Accomplished

You now know how to:

- Rent and configure a cloud GPU for LLM inference
- Install and run vLLM on RunPod
- Serve GPT-OSS 120B (120 billion parameters) via API
- Interact with the model using curl and the OpenAI Python SDK
- Build streaming and batch inference applications
- Connect Cline as your AI coding assistant via RunPod proxy
- Fine-tune models with QLoRA
- Switch between different models on the same GPU

**Next steps**:
- Try the [DGX Spark version](../README.md) if you get a Spark
- Fine-tune a model on your own data
- Experiment with different models for different tasks
- Build applications using the OpenAI-compatible API

---

## Architecture Diagram (RunPod)

```
┌──────────────────────────────────────────────────────────────┐
│                    RunPod Cloud Pod                           │
│                   (A100 80GB VRAM)                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  vLLM Server (Port 8000)                            │    │
│  │  ├─ FastAPI: /v1/models, /v1/chat/completions      │    │
│  │  ├─ Continuous batching scheduler                   │    │
│  │  └─ PagedAttention KV cache manager                 │    │
│  └───────────────────┬─────────────────────────────────┘    │
│                      │                                       │
│  ┌───────────────────▼─────────────────────────────────┐    │
│  │  GPT-OSS 120B (mxfp4, ~65GB)                        │    │
│  │  ├─ Harmony tokenizer (chat template)               │    │
│  │  └─ Reasoning: analysis/commentary/final            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Persistent Volume (/workspace, 100GB)              │    │
│  │  ├─ /workspace/hf-cache/ (model weights)            │    │
│  │  └─ /workspace/dgx-spark-ai/ (project code)        │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ HTTPS (RunPod Proxy)
                         │ https://<pod-id>-8000.proxy.runpod.net
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Your Local Machine                                          │
│  ├─ curl / Python scripts (OpenAI SDK)                      │
│  ├─ Cline CLI (AI coding assistant)                         │
│  └─ Any OpenAI-compatible client                            │
└──────────────────────────────────────────────────────────────┘
```

---

*Adapted from the [DGX Spark curriculum](../README.md) for cloud GPU usage on RunPod.*
