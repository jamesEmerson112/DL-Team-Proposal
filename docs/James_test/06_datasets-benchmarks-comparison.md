# Datasets & Benchmarks Comparison

## Datasets

| Dataset | nanoGPT | llm.c | nanochat |
|---|---|---|---|
| OpenWebText | Primary | - | - |
| FineWeb / FineWeb-EDU | - | Primary | Primary |
| NVIDIA ClimbMix | - | - | Alt (speedruns) |
| Shakespeare | Quick-start | - | - |
| TinyShakespeare | - | Quick-start | - |
| TinyStories | - | Quick-start | - |
| SmolTalk (460K convos) | - | - | SFT stage |
| MMLU aux-train (100K) | - | - | SFT stage |
| GSM8K (8K math) | - | - | SFT + RL stages |

- nanoGPT uses the older **OpenWebText**; llm.c and nanochat use **FineWeb/FineWeb-EDU** (newer, larger, higher quality)
- nanochat is the only one with SFT/RL-specific datasets since it's the only one doing post-training

## Benchmarks

| Benchmark | nanoGPT | llm.c | nanochat |
|---|---|---|---|
| Validation loss | Yes | Yes | Yes |
| HellaSwag | - | Yes (native) | Yes (via DCLM CORE) |
| MMLU | - | - | Yes |
| ARC | - | - | Yes |
| GSM8K | - | - | Yes |
| HumanEval | - | - | Yes |
| TriviaQA | - | - | Yes |
| DCLM CORE (22-task) | - | - | Yes |

- nanoGPT only tracks validation loss — no formal benchmarks
- llm.c adds HellaSwag (commonsense reasoning, 4-choice, 10K sentences)
- nanochat has the most comprehensive eval with the 22-task DCLM CORE score

## Conclusion

nanochat has the most complete coverage on both datasets and evaluation. It's also the only project that covers the full training pipeline (pretrain → SFT → RLHF).
