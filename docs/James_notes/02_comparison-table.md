# Framework Comparison: nanoGPT vs llm.c vs nanochat

| Feature | nanoGPT | llm.c | nanochat |
|---|---|---|---|
| **Language** | Python / PyTorch | C / CUDA | Python / PyTorch |
| **Dependencies** | Heavy (PyTorch) | Minimal (none) | Heavy (PyTorch) |
| **Performance** | Good (torch.compile) | ~7% faster than PyTorch | Good (Flash Attention 3) |
| **Parameters** | 124M–1.5B (GPT-2) | 124M–1.5B (GPT-2/3) | 560M (standard) |
| **Context window** | 1024 (GPT-2) | 1024 default, configurable | 2048 (sliding window attention) |
| **Training stages** | Pretraining only | Pretraining only | Pretrain + SFT + RLHF |
| **Custom data** | Yes, any text | Yes, any text via scripts | Yes, including synthetic data |
| **Built-in benchmarks** | None | HellaSwag, MMLU | ARC, MMLU, GSM8K, HumanEval |
| **Distributed** | DDP (PyTorch) | MPI + NCCL | Yes |
| **Inference / UI** | No | No | Yes (KV cache + web UI) |
| **Cost to train** | Varies | Varies | ~$100 (8xH100, 4hrs) |
| **Ease of use** | Low barrier | High barrier (C/CUDA) | Low–Medium barrier |
| **Status** | Deprecated | Active | Active |
| **Best for** | Quick experiments | Low-level GPU learning | Full ChatGPT-like pipeline |
