# Note 18 — LLM Parameter Anatomy

> Where do the parameters live in a language model, and how much space does each component take?

## PG Baseline Model (17M params)

Config: 9 layers, 512 dims, 8 heads, 4 KV heads (GQA), 1024 vocab, tied embeddings, MLP mult 2.

### Parameter Breakdown

| Component | Per Layer | Total (9 layers) | % of Model |
|---|---|---|---|
| **MLP** | 1,048,576 | 9,437,184 | **55.3%** |
| **Attention** | 786,440 | 7,077,960 | **41.5%** |
| **Embedding** | — | 524,288 | **3.1%** |
| Skip weights + scalars | — | ~2,500 | 0.01% |
| **Total** | — | **~17,042,000** | 100% |

```
████████████████████████████████████████████  55.3%  MLP
██████████████████████████████████           41.5%  Attention
██                                            3.1%  Embedding
```

### MLP (55.3% of model) — the largest component

Each layer has two linear projections with ReLU² activation:

```
x ──► Up projection (512 → 1024)    524,288 params
      ──► ReLU² activation           0 params (just math)
      ──► Down projection (1024 → 512)  524,288 params ──► out
```

Total per layer: 1,048,576 params. MLP mult=2 means hidden dim is 2× model dim.

**Why it matters:** Over half the model lives here. Activation choice (ReLU² vs LeakyReLU² vs SwiGLU) directly affects how well these params learn. Quantizing MLP layers aggressively (int6 vs int8) has the biggest impact on compressed size.

### Attention (41.5% of model) — second largest

Each layer projects input into Q, K, V, computes attention, then projects back:

```
x ──► Q projection (512 → 512)     262,144 params   (8 heads × 64 dim)
      K projection (512 → 256)     131,072 params   (4 KV heads × 64 dim, GQA)
      V projection (512 → 256)     131,072 params   (4 KV heads × 64 dim, GQA)
      ──► RMSNorm, RoPE, SDPA       0 params (norms have no learnable weights here)
      ──► Out projection (512 → 512) 262,144 params
      + q_gain scalars                     8 params ──► out
```

Total per layer: 786,440 params.

**GQA savings:** Standard MHA would need 8 KV heads (512 dim each for K and V = 524,288 params). GQA uses 4 KV heads (262,144 params) — saves 262,144 params per layer (50% KV savings).

**Why it matters:** Attention techniques (gated attention, XSA, QK-gain) modify how these params interact but don't necessarily add many new params. Headwise gated attention adds only 8 params/layer (the gate scalars).

### Embedding (3.1% of model) — smallest but most sensitive

```
Token ID ──► Embedding table (1024 × 512)    524,288 params
             (also reused as output head via tied embeddings)
```

**Tied embeddings** means the same 524K params serve double duty:
- Input: token ID → 512-dim vector
- Output: 512-dim vector → 1024 logits (next token prediction)

**Why it matters:** Despite being only 3% of params, the embedding is the most sensitive tensor to quantize (Renier Velazco's finding). It's pulling double duty — int8 quantization degrades BPB by ~0.007, while keeping it in fp16 costs only ~500KB extra but reduces degradation to ~0.0005.

### What changes with SP8192?

Switching vocab from 1024 to 8192 changes the embedding:

| Vocab | Embedding params | % of model | Size in fp16 |
|---|---|---|---|
| 1024 | 524,288 | 3.1% | 1.0 MB |
| 4096 | 2,097,152 | 11.2% | 4.0 MB |
| 8192 | 4,194,304 | 19.7% | 8.0 MB |

Bigger vocab = each token carries more information → fewer tokens needed → better BPB. But the embedding table grows, eating into the 16 MB budget. Top leaderboard entries (SP8192) deal with this through aggressive quantization of other layers.

## Full Forward Pass Pipeline

```
"The quick brown fox"
        │
   TOKENIZER (SentencePiece, 1024 vocab)         ← not part of model weights
   "The"→42, "quick"→817, ...
        │
   EMBEDDING TABLE  [3.1% of params]             ← most sensitive to quantize
   token ID → 512-dim vector
        │
   RMSNorm
        │
  ┌─ ENCODER BLOCKS 0-3 ──────────────────┐
  │  Attention [41.5%] + MLP [55.3%]       │     ← bulk of the model
  │  Each block saves a skip tensor         │
  └────────────────────────────────────────┘
        │
  ┌─ DECODER BLOCKS 4-8 ──────────────────┐
  │  Attention + MLP                        │
  │  + skip connections from encoder (U-Net)│
  └────────────────────────────────────────┘
        │
   FINAL RMSNorm
        │
   LM HEAD (reuses embedding table)              ← tied = 0 extra params
   512-dim → 1024 logits
        │
   SOFTCAP (tanh capping)
        │
   CROSS-ENTROPY LOSS → backprop
```

## Implications for Parameter Golf

1. **MLP is where most params live** — activation function and MLP width have outsized impact
2. **Attention is the "intelligence"** — techniques like gated attention, depth recurrence, and QK-gain modify attention behavior with minimal param cost
3. **Embedding is tiny but critical** — keep it in fp16 during quantization (Velazco's insight)
4. **Quantization strategy should match component sensitivity:**
   - Embedding: fp16 (most sensitive, only 3% of params)
   - Early/late layers: int8 (sensitive, bookend the model)
   - Middle layers: int6 (redundant, tolerate lower precision)

## Cross-Reference

- `16_pg-leaderboard-techniques.md` — technique deep dives (SP8192, depth recurrence, etc.)
- `17_pg-leaderboard-annotated.md` — all leaderboard entries with components tagged
- `parameter-golf/train_gpt.py` lines 765-843 — GPT class and forward pass
