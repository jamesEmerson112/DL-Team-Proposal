# llm.c — Research Notes

**Author:** Andrej Karpathy
**Repo:** github.com/karpathy/llm.c
**License:** MIT
**Status:** Active

## Overview

llm.c implements LLM training in raw C/CUDA with zero dependency on PyTorch or Python. The entire training stack is ~5,000 lines of C/CUDA. Its goal is to reproduce the GPT-2 and GPT-3 model miniseries from first principles in a minimal, readable codebase.

Ships with a parallel PyTorch reference (`train_gpt2.py`, described as "a slightly tweaked nanoGPT") for validation.

## Design Philosophy

- Simplicity and readability over marginal performance gains
- PRs that improve perf but add substantial complexity are rejected
- `dev/` folder for experimental kernels; root code stays clean
- Karpathy only maintains C and CUDA — community ports (Rust, Zig, Go) are separate repos

## Architecture

| File | Purpose | LOC |
|---|---|---|
| `train_gpt2.cu` | Main CUDA implementation (model + training loop) | ~2,000–3,000 |
| `train_gpt2.c` | CPU fp32 reference (educational) | ~1,000 |
| `train_gpt2fp32.cu` | Pure CUDA fp32, no cublas/cudnn (course material) | — |
| `train_gpt2.py` | PyTorch reference implementation | — |
| `common.h` | Shared utilities | ~300 |
| `tokenizer.h` | Tokenizer support | ~100 |

**Layer implementations** (CUDA headers):
- `encoder.cuh` — encoder forward/backward
- `layernorm.cuh` — layernorm, residual, fused residual
- `matmul.cuh` — matmul via cublasLt, GELU forward/backward
- `attention.cuh` — attention forward/backward (with cuDNN flash attention option)

Forward and backward passes are implemented manually, strung into a large manual loop. Optimizer is **AdamW** with cosine decay + warmup and gradient clipping.

## Key CUDA Optimizations

- **Packed128**: Forces 128-bit `LDG.128`/`STS.128` instructions, maximizing memory bandwidth
- **FusedClassifier**: Doesn't materialize full logits tensor — loss evaluated only at label index
- **Gradient elimination**: Deletes unnecessary backward pass memory → **16.6 GiB** vs PyTorch's **37.2 GiB** (~55% reduction)
- **Flash Attention via cuDNN**: Optional (`USE_CUDNN=1`), increases compile from seconds to ~1 minute
- **cuBLAS/cublasLt**: Used by default for matrix multiplication

## Performance

**vs PyTorch (May 2024):**
- ~7% faster than PyTorch Nightly (with all optimizations: torch.compile, flash attention, bf16)
- ~46% faster than PyTorch 2.3.0 stable
- 55% less peak memory

**GPT-2 124M reproduction (8x A100 80GB):**
- ~90 minutes, ~$20
- MFU: ~60% (A100 80GB SXM), 48–49% (A100 40GB PCIe)
- Throughput: ~178K tok/s (single A100 40GB)
- HellaSwag: 29.9% (beats original GPT-2 at 29.4%)

**GPT-2 1.6B reproduction (8x H100):**
- 24 hours, $672, 32,000 steps
- 33.6B tokens, ~381K tok/s, MFU ~50% bf16
- HellaSwag: 51% at 32K steps, ~61% at 400K steps

## Supported Models

| Model | Flag | Params | Seq Length |
|---|---|---|---|
| GPT-2 Small | `d12` | 124M | 1024 |
| GPT-2 Medium | `d24` | 350M | 1024 |
| GPT-2 Large | `d36` | 774M | 1024 |
| GPT-2 XL | `d48` | 1,558M | 1024 |
| GPT-3 (various) | `gpt3:cX` | varies | 2048 |
| Llama3 | reference impl | varies | varies |

## Distributed Training (MPI + NCCL)

- **NCCL** for GPU-to-GPU collectives (allreduce)
- **OpenMPI** for process management and NCCL ID exchange
- **ZeRO-1** (`-z 1`): Optimizer state sharding across GPUs
- Alt initialization: shared filesystem or TCP sockets
- Multi-node scripts in `scripts/multi_node/`

## Dependencies and Build

**Hard requirements:** C compiler + NVIDIA CUDA toolkit

**Optional:**
- cuDNN (flash attention)
- OpenMPI + NCCL (multi-GPU)
- PyTorch (only for reference impl and data prep)

```bash
# CPU fp32 reference
make train_gpt2

# GPU (main, bf16 mixed precision)
make train_gpt2cu

# GPU with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Tests
make test_gpt2cu
```

Makefile auto-detects GPU compute capability via `nvidia-smi`.

## Datasets

| Dataset | Use Case |
|---|---|
| **FineWeb** (primary) | 15T-token dataset from 96 Common Crawl snapshots; 124M uses 10B tokens, 1.6B uses FineWeb-Edu 100B |
| TinyShakespeare | Quick-start small dataset |
| TinyStories | Smaller-scale experiments |

**Data format:** Custom `.bin` files with uint16 token streams and 1024-byte headers. Prep scripts in `dev/data/`.

## Benchmarks

- **HellaSwag**: Natively supported (`-h 1` flag). Commonsense reasoning, 4-choice sentence completion, 10K sentences.
- **MMLU**: Not natively supported (available in nanochat instead)

## Key Files

```
llm.c/
  train_gpt2.cu        # Main CUDA training
  train_gpt2.c         # CPU fp32 reference
  train_gpt2.py        # PyTorch reference
  train_gpt2fp32.cu    # Pure CUDA fp32 (educational)
  test_gpt2.c/cu       # Unit tests
  profile_gpt2.cu      # Profiling harness
  common.h / tokenizer.h
  Makefile
  dev/
    cuda/              # Educational kernel library (increasing complexity)
    data/              # Dataset preparation scripts
  scripts/
    multi_node/        # MPI multi-node scripts
```

The `dev/cuda/` directory is notable: each file shows kernel versions of increasing complexity/speed, with CPU reference → GPU implementations compared for correctness.

## Evolutionary Context

Per Karpathy: "First I wrote [nanoGPT] to teach people the basics of training GPTs. Then it became a target and baseline for my port to direct C/CUDA [llm.c]." The community then created **modded-nanoGPT** (optimized GPT-2 124M from 45 min → 3 min), which inspired **nanochat** to extend the scope to the full ChatGPT pipeline.
