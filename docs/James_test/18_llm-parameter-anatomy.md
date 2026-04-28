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

## The Brain Analogy

A language model's four components map cleanly to how the brain processes language:

```
Embedding  = Dictionary / Language Center (Wernicke's area)
             Word ↔ meaning translation. "dog" → [mammal, pet, four legs, ...]
             Only stores what each word means as a starting point.
             Doesn't know facts, grammar, or reasoning.

Attention  = Working Memory / Focus System (Prefrontal cortex)
             Connects "it" in "The dog chased the cat and it was tired"
             back to the right noun. Handles relationships BETWEEN tokens —
             syntax, coreference, context. The "thinking about what I've
             read so far" mechanism. This is the inference engine.

MLP        = Long-Term Memory / Knowledge Store (Cortex)
             "Paris is the capital of France" lives in MLP weights.
             Processes each token independently — enriches representations
             with stored knowledge. Research shows you can locate specific
             factual associations in specific MLP layers.

Skip Wts   = Corpus Callosum (connects early and late processing)
             U-Net shortcuts from encoder to decoder layers.
```

They alternate every layer: **attention gathers context** (what's relevant?), **MLP enriches with knowledge** (what do I know about this?), repeat 9 times, each pass building deeper understanding.

This is why quantizing MLP aggressively (int6) risks losing stored knowledge, while quantizing attention risks losing the model's ability to reason about context. And the embedding — the dictionary — must stay precise because a corrupted word meaning propagates through every layer.

## Implications for Parameter Golf

1. **MLP is long-term memory** — activation function and MLP width control how much knowledge the model can store. Over half the params live here.
2. **Attention is working memory** — techniques like gated attention, depth recurrence, and QK-gain modify reasoning behavior with minimal param cost
3. **Embedding is the dictionary** — tiny but most sensitive to quantize. Keep in fp16 (Velazco's insight).
4. **Quantization strategy should match component sensitivity:**
   - Embedding: fp16 (most sensitive, the dictionary)
   - Early/late layers: int8 (sensitive, set up and finalize representations)
   - Middle layers: int6 (most redundant, tolerate lower precision)

## SP8192 Combo Slim — Run 11 (16.4M params)

Our best submittable config. Same 9-layer U-Net architecture, but dim reduced 512→448 to fit SP8192 embedding under 16 MB.

Config: 9 layers, 448 dims, 8 heads, 4 KV heads (GQA), 8192 vocab, tied embeddings, MLP mult 2, headwise gated attn, LeakyReLU², QK-Gain 5.0.

### Parameter Breakdown

| Component | Per Layer | Total (9 layers) | % of Model |
|---|---|---|---|
| **MLP** | 802,816 | 7,225,344 | **44.2%** |
| **Attention** | 605,704 | 5,451,336 | **33.3%** |
| **Embedding** | — | 3,670,016 | **22.4%** |
| Skip weights + scalars | — | ~1,800 | 0.01% |
| **Total** | — | **~16,348,500** | 100% |

```
█████████████████████████████████████  44.2%  MLP
████████████████████████████           33.3%  Attention
██████████████████                     22.4%  Embedding
```

### MLP (dim=448)

```
x ──► Up projection (448 → 896)    401,408 params
      ──► LeakyReLU(0.5)² activation    0 params
      ──► Down projection (896 → 448)  401,408 params ──► out
```

### Attention with headwise gate (dim=448)

```
x ──► Q projection (448 → 456)    204,288 params   (8 heads × 56 dim + 8 gate dims)
      K projection (448 → 224)    100,352 params   (4 KV heads × 56 dim, GQA)
      V projection (448 → 224)    100,352 params   (4 KV heads × 56 dim, GQA)
      ──► RMSNorm, RoPE, SDPA, sigmoid gate    0 learnable params
      ──► Out projection (448 → 448)  200,704 params
      + q_gain scalars                      8 params ──► out
```

head_dim = 448/8 = 56 (vs baseline's 64). Headwise gate widens Q projection by 8 dims (72 extra params across 9 layers — negligible).

### What changed from baseline

| Metric | Baseline (SP1024, 512d) | Run 11 (SP8192, 448d) | Delta |
|---|---|---|---|
| Embedding % | 3.1% | 22.4% | +19.3% — 8× vocab makes embedding dominant |
| MLP % | 55.3% | 44.2% | -11.1% — dim reduction shrinks MLP |
| Attention % | 41.5% | 33.3% | -8.2% — dim reduction shrinks attention |
| Total params | 17.04M | 16.36M | -0.68M — smaller model, bigger vocab |
| val_bpb | 1.2244 (baseline) | **1.2077** | **-0.0167** |

The 8× vocab adds 3.15M embedding params but dim reduction (512→448) saves 3.83M from MLP+attention. Net: 0.68M fewer params, yet 0.017 better BPB — the vocab upgrade more than compensates.

## SP8192 Combo — Run 10 (20.8M params)

The "what if GPTQ freed budget" config. Same techniques as Run 11 but dim=512 (no reduction needed). Produces **1.1872 BPB** but 19.41 MB int8+zlib — 3.41 MB over budget without GPTQ.

Config: 9 layers, 512 dims, 8 heads, 4 KV heads (GQA), 8192 vocab, tied embeddings, MLP mult 2, headwise gated attn, LeakyReLU², QK-Gain 5.0.

### Parameter Breakdown

| Component | Per Layer | Total (9 layers) | % of Model |
|---|---|---|---|
| **MLP** | 1,048,576 | 9,437,184 | **45.4%** |
| **Attention** | 790,536 | 7,114,824 | **34.3%** |
| **Embedding** | — | 4,194,304 | **20.2%** |
| Skip weights + scalars | — | ~2,050 | 0.01% |
| **Total** | — | **~20,748,000** | 100% |

### Where the extra 4.4M params come from (vs Run 11)

| Component | Run 11 (448d) | Run 10 (512d) | Extra |
|---|---|---|---|
| MLP (9 layers) | 7,225,344 | 9,437,184 | +2,211,840 |
| Attention (9 layers) | 5,451,336 | 7,114,824 | +1,663,488 |
| Embedding | 3,670,016 | 4,194,304 | +524,288 |
| **Total** | **~16,348,500** | **~20,748,000** | **~+4,400,000** |

Each dim step (448→512) costs ~4.4M params. MLP accounts for half the increase because it has 2× multiplier on dim.

### Elementwise gated attention variant

For reference, elementwise gated attention doubles the Q projection width (dim → 2×dim for Q+gate logits), adding 262,144 params per layer at dim=512. Run 3 (SP1024, elementwise, dim=512) measured 19.42M params — the +2.36M over baseline comes entirely from the wider Q projection.

## Quantization Anatomy

How each compression scheme affects the artifact, using actual run data from Run 11's config (headwise dim=448, SP8192).

### Aggregate comparison

| Scheme | Artifact size | Headroom | BPB gap vs raw | Source |
|---|---|---|---|---|
| Raw fp16 | ~31.2 MB | — | 0 | Theoretical |
| int8+zlib | 15.35 MB | 0.65 MB | +0.0019 | Run 11 (8×H100) |
| GPTQ int6+brotli (v1) | 10.50 MB | 5.50 MB | +0.1062 roundtrip | GPTQ smoke test |
| GPTQ int6+brotli + TTT | 10.50 MB | 5.50 MB | +0.0541 | GPTQ smoke + TTT |
| Kevin Clark target (rank 5) | ~10 MB | ~6 MB | +0.012 | Rank 5 README |

**Size is excellent** (10.50 MB). **BPB gap is 10× worse than expected** — our v1 GPTQ used the rank 9 approach (5-percentile clip search, AR self-gen, `inference_mode`). Kevin Clark's approach (k×std single pass, training data calibration, `no_grad`) achieves +0.012 gap. The gap is a code quality issue, not a fundamental limit.

### Per-component sensitivity (from leaderboard analysis)

| Component | % of params | Quant tolerance | Notes |
|---|---|---|---|
| **Embedding** | 22.4% | Low — keep int8 or fp16 | Most sensitive tensor. Renier Velazco: fp16 embedding reduces degradation ~0.007→~0.0005. Kevin Clark quantizes embedding separately with higher clip_k (20.0 vs 12.85). |
| **MLP** | 44.2% | High — int6 works well | Low-entropy weights (ReLU²/LeakyReLU² creates sparsity). Largest component, so int6 here saves the most bytes. |
| **Attention** | 33.3% | Medium | Q projection most sensitive (carries gate logits in headwise). K/V projections tolerate int6. |
| **Early blocks (0-1)** | ~22% | Low | Hessian traces 30× larger than late blocks (Robby Sneiderman, rank 4). These set up representations. |
| **Late blocks (7-8)** | ~22% | High | Most redundant, tolerate aggressive quantization. |

### GPTQ bugs and lessons learned

1. **`inference_mode` poisons Rotary cache** — Hessian collection with `torch.inference_mode()` creates permanently tainted "inference tensors" that leaked into the Rotary cos/sin cache, crashing TTT. Fix: use `torch.no_grad()` + `.clone()` on Rotary output.
2. **Percentile clip search is 5× slower** — Rank 9/10 approach runs GPTQ 5× per matrix. Kevin Clark's `k × std(row)` single pass achieves the same compressed size with Cholesky error compensation.
3. **int7 with high k beats int6** — `clip_range=63` (int7) with `k=12.85` produces the same compressed size as int6 but with less clipping error, because brotli compresses the narrower value range just as efficiently.
4. **Training data calibration beats AR self-gen** — AR self-gen takes ~120s, training data takes ~5s. Kevin Clark uses 64 batches from training data. Quality is at least as good.

## Budget Math — What GPTQ Unlocks

GPTQ int6+brotli compresses to ~68% of int8+zlib size (observed: 10.50/15.35 = 0.684×). Applying this ratio to other configs:

| Config | Params | int8+zlib | GPTQ est. | Fits 16 MB? |
|---|---|---|---|---|
| Run 11 (headwise dim=448) | 16.4M | 15.35 MB | **~10.5 MB** | Yes (5.5 MB free) |
| Run 10 (headwise dim=512) | 20.8M | 19.41 MB | **~13.3 MB** | **Yes** |
| E1 (elementwise dim=448) | ~18.1M | 16.67 MB | **~11.4 MB** | Yes |
| 11 layers, dim=448 | ~19.5M | ~18.0 MB | **~12.3 MB** | Yes |
| MLP 3×, dim=448 | ~19.9M | ~18.8 MB | **~12.9 MB** | Yes |

With 5.5 MB headroom from GPTQ, the following upgrades become budget-legal:

- **dim=512** — restore the 0.020 BPB lost to dim reduction (Run 10: 1.1872 vs Run 11: 1.2077)
- **11 layers** — add 2 layers for more depth (common in PG ranks 12-21)
- **MLP 3×** — wider MLP for better token representation (common in ranks 5-22)
- **Elementwise gated attn** — doubles Q projection for per-dim gating (E1: best 2×H100 BPB, 1.2338)

The biggest bang-for-buck is **dim=512**: it recovers 0.020 BPB from a single config change and still leaves ~2.7 MB headroom for further upgrades.

## How Our Brain Learns — The Training Pipeline as Neural Development

This traces what happens when our Run 11 brain (9L, dim=448, SP8192) processes the sentence **"The dog chased the cat and it was tired"** during training — one complete forward pass + backward pass, explained as a developing brain learning language.

### Phase 1: Hearing the Words (Embedding)

```
 SENSORY INPUT — Wernicke's Area (Dictionary)
 ══════════════════════════════════════════════

 "The dog chased the cat and it was tired"
   │    │     │     │   │   │   │  │    │
   ▼    ▼     ▼     ▼   ▼   ▼   ▼  ▼    ▼
 ┌────────────────────────────────────────────┐
 │         EMBEDDING TABLE (22.4%)            │
 │         3.7M synapses                      │
 │                                            │
 │  "The" → [0.12, -0.34, 0.81, ... ×448]   │
 │  "dog" → [0.77,  0.23, 0.05, ... ×448]   │
 │  "it"  → [0.41, -0.15, 0.62, ... ×448]   │
 │                                            │
 │  Each word becomes a 448-dimensional       │
 │  "sensation" — a raw, unprocessed feeling  │
 │  of what the word means in isolation.      │
 └────────────────────────────────────────────┘
          │
          ▼
     9 tokens, each a 448-dim vector
     (the brain has "heard" the words but
      doesn't understand the sentence yet)
```

**Brain analogy:** This is like a newborn hearing speech — the auditory cortex activates for each sound, mapping it to a basic representation. "Dog" triggers a vague cluster of associated neurons, but there's no understanding of *which* dog, *what it did*, or *why it matters*. That comes next.

### Phase 2: Building Understanding (Encoder — Layers 0-3)

The encoder is like the **ventral stream** in neuroscience — the "what pathway" that builds increasingly abstract representations. Each layer alternates between two operations: attention (connecting words) and MLP (enriching with knowledge).

```
 ENCODER — Ventral Stream ("What Pathway")
 ══════════════════════════════════════════

 ┌─ LAYER 0 ──────────────────────────────────────────────────────┐
 │                                                                │
 │  ATTENTION (focus): "What relates to what?"                    │
 │  ┌──────────────────────────────────────────────────────────┐  │
 │  │  "The" ←──── looks at ────► "dog"    (article + noun)   │  │
 │  │  "the" ←──── looks at ────► "cat"    (article + noun)   │  │
 │  │  "chased" ← looks at ──► "dog","cat" (verb + arguments) │  │
 │  │                                                          │  │
 │  │  Each of 8 attention heads focuses on a different        │  │
 │  │  relationship type: syntax, proximity, semantics...      │  │
 │  │                                                          │  │
 │  │  Headwise gate: sigmoid decides how much each head's     │  │
 │  │  finding matters. "Head 3 found something useful → let   │  │
 │  │  it through. Head 7 found noise → suppress it."          │  │
 │  └──────────────────────────────────────────────────────────┘  │
 │       │                                                        │
 │       ▼                                                        │
 │  MLP (knowledge): "What do I know about this?"                 │
 │  ┌──────────────────────────────────────────────────────────┐  │
 │  │  448 → 896 → 448  (expand, think, compress)              │  │
 │  │                                                          │  │
 │  │  "dog" + context of "chased" →                           │  │
 │  │     enriched "dog" now contains: [agent, animate,        │  │
 │  │     predator-role, past-tense-action, ...]               │  │
 │  │                                                          │  │
 │  │  LeakyReLU²: the activation function. Neurons that fire  │  │
 │  │  weakly still contribute (leaky), and strong signals     │  │
 │  │  are amplified quadratically (²). Like neural gain       │  │
 │  │  control — the brain turns up volume on important        │  │
 │  │  signals and keeps a whisper of everything else.         │  │
 │  └──────────────────────────────────────────────────────────┘  │
 │       │                                                        │
 │       ├──── SAVE skip tensor (for decoder layer 8) ◄──┐       │
 │       │     Like forming an episodic memory snapshot    │       │
 │       ▼                                                 │       │
 └─────────────────────────────────────────────────────────┘       │
                                                                   │
 ┌─ LAYER 1 ──────────────────────────────────────────────────┐   │
 │  Attention: deeper relationships emerge                     │   │
 │    "it" starts to weakly attend to both "dog" and "cat"    │   │
 │    (hasn't resolved the ambiguity yet)                      │   │
 │  MLP: more abstract enrichment                              │   │
 │    "chased" → [pursuit, predator-prey, motion-verb, ...]   │   │
 │       │                                                     │   │
 │       ├──── SAVE skip tensor (for decoder layer 7)          │   │
 │       ▼                                                     │   │
 └─────────────────────────────────────────────────────────────┘   │
                                                                   │
 ┌─ LAYER 2 ──────────────────────────────────────────────────┐   │
 │  Attention: multi-hop reasoning begins                      │   │
 │    "it" attends to "dog" AND "chased" together —            │   │
 │    starting to figure out the agent of the action           │   │
 │  MLP: factual associations activate                         │   │
 │    "tired" → [state-change, consequence, animate-only, ...] │   │
 │       │                                                     │   │
 │       ├──── SAVE skip tensor (for decoder layer 6)          │   │
 │       ▼                                                     │   │
 └─────────────────────────────────────────────────────────────┘   │
                                                                   │
 ┌─ LAYER 3 ──────────────────────────────────────────────────┐   │
 │  Attention: abstract patterns crystallize                   │   │
 │    "it" now strongly attends to "dog" (the chaser gets     │   │
 │    tired, not the chasee — learned from training data)      │   │
 │  MLP: highest-level encoder knowledge                       │   │
 │    sentence structure fully parsed                          │   │
 │       │                                                     │   │
 │       ├──── SAVE skip tensor (for decoder layer 5)          │   │
 │       ▼                                                     │   │
 └─────────────────────────────────────────────────────────────┘   │
                                                                   │
         At this point, the brain has a deep understanding         │
         of the sentence. But understanding isn't enough —         │
         it needs to produce an output (predict the next word).    │
         That's the decoder's job.                                 │
                                                                   │
                                        skip connections ──────────┘
```

**Brain analogy:** This is like reading a sentence carefully for the first time. Your brain processes it bottom-up — first recognizing individual words (layer 0), then parsing syntax (layer 1), then resolving references (layer 2), then grasping full meaning (layer 3). Each layer's output is a richer representation. The skip connections are like **episodic memory** — snapshots of your understanding at each stage, preserved for later use.

### Phase 3: Generating Output (Decoder — Layers 4-8)

The decoder is like the **dorsal stream** — the "how pathway" that transforms understanding into action (predicting the next token). It receives skip connections from the encoder, like recalling earlier stages of comprehension.

```
 DECODER — Dorsal Stream ("How Pathway")
 ═════════════════════════════════════════

 ┌─ LAYER 4 ──────────────────────────────────────────────────┐
 │  (no skip connection — this is the U-Net "bottleneck")     │
 │  Deepest abstract processing — the "aha moment"            │
 │  Attention + MLP refine the representation further         │
 │       │                                                     │
 │       ▼                                                     │
 └─────────────────────────────────────────────────────────────┘

 ┌─ LAYER 5 ──────────────────────────────────────────────────┐
 │  + SKIP from Layer 3 (high-level encoder snapshot)         │
 │  ┌────────────────────────────────────────────────────┐    │
 │  │  current representation ──┐                        │    │
 │  │                           ├─► BLEND (learned wt)   │    │
 │  │  layer 3 skip tensor ────┘                         │    │
 │  │                                                    │    │
 │  │  Like recalling your first careful reading while   │    │
 │  │  also having the deeper layer-4 "aha" insight.     │    │
 │  │  The skip connection prevents forgetting the        │    │
 │  │  original parse as reasoning gets more abstract.   │    │
 │  └────────────────────────────────────────────────────┘    │
 │  Attention + MLP continue refining                         │
 │       │                                                     │
 │       ▼                                                     │
 └─────────────────────────────────────────────────────────────┘

 ┌─ LAYER 6 ── + SKIP from Layer 2 ──────────────────────────┐
 │  Recalls mid-level parsing (multi-hop reasoning stage)     │
 │  Starts converting abstract understanding → concrete output│
 │       ▼                                                     │
 └─────────────────────────────────────────────────────────────┘

 ┌─ LAYER 7 ── + SKIP from Layer 1 ──────────────────────────┐
 │  Recalls syntactic structure from early processing         │
 │  Output representations becoming more "word-like"          │
 │       ▼                                                     │
 └─────────────────────────────────────────────────────────────┘

 ┌─ LAYER 8 ── + SKIP from Layer 0 ──────────────────────────┐
 │  Recalls raw word-level features from first layer          │
 │  Final refinement: abstract meaning → concrete word choice │
 │       │                                                     │
 │       ▼                                                     │
 │  Output: 9 refined 448-dim vectors, one per token          │
 └─────────────────────────────────────────────────────────────┘
```

**Brain analogy:** The decoder is like **formulating a response** after comprehending a sentence. You understood it (encoder), and now you're producing language (decoder). The skip connections are crucial — they prevent the "tip of the tongue" phenomenon where deep understanding can't connect back to specific words. Layer 8 receiving Layer 0's skip is like your speech production center accessing your raw phonological memory to pick the right word.

### Phase 4: The Prediction (LM Head)

```
 PREDICTION — Broca's Area (Speech Production)
 ═══════════════════════════════════════════════

 Layer 8 output for position 9 ("tired"):
 [0.83, -0.21, 0.54, ... ×448]
          │
          ▼
 ┌────────────────────────────────────────────┐
 │  LM HEAD (reuses Embedding table)          │
 │  448-dim vector → 8,192 logits             │
 │                                            │
 │  "."     → logit 4.2   (highest!)         │
 │  ","     → logit 3.1                       │
 │  "and"   → logit 1.8                       │
 │  "dog"   → logit 0.3                       │
 │  "Paris" → logit -2.1  (irrelevant)        │
 │                                            │
 │  The brain predicts: next word is "."      │
 │  (The sentence is complete.)               │
 └────────────────────────────────────────────┘
          │
          ▼
     softmax → probability distribution
     cross-entropy loss: how wrong were we?
```

### Phase 5: Learning from Mistakes (Backpropagation)

This is where the brain actually **grows**. Everything above was one forward pass — the brain reading and predicting. Now the error signal flows backward, strengthening or weakening every synapse.

```
 BACKPROPAGATION — Synaptic Plasticity
 ══════════════════════════════════════

 The true next word was "." and we predicted "." → small loss
 But at position 7, we predicted "is" and truth was "was" → larger loss

 Error signal flows BACKWARD through every layer:

 Loss = 2.14 (how wrong the brain was overall)
          │
          ▼
 Layer 8 ◄── "Your final word choice was slightly off.
              Adjust your 605K attention + 803K MLP synapses."
          │
          ▼
 Layer 7 ◄── "Your syntax reconstruction was good but
              the tense handling needs work."
          │
          ▼
    ... (error flows through every layer) ...
          │
          ▼
 Layer 0 ◄── "Your initial word groupings were fine.
              Tiny adjustments only."
          │
          ▼
 Embedding ◄── "Your definition of 'was' needs to
                encode tense information more strongly."


 LEARNING RULE (simplified):

   For each synapse (weight) in the brain:
   ┌─────────────────────────────────────────────────┐
   │                                                 │
   │  gradient = "how much did THIS synapse          │
   │              contribute to the mistake?"         │
   │                                                 │
   │  new_weight = old_weight - learning_rate × grad │
   │                                                 │
   │  Muon optimizer: also considers the geometry    │
   │  of the loss landscape (not just steepest       │
   │  descent, but the most efficient direction)     │
   │                                                 │
   └─────────────────────────────────────────────────┘

 This happens for ALL 16.4M synapses simultaneously.
 One forward + backward pass ≈ 1 training step.
 Run 11 does ~11,000 steps in 10 minutes on 8×H100.
```

**Brain analogy:** This is **Hebbian learning** — "neurons that fire together wire together." But backprop is more precise: it knows exactly *which* synapses caused *which* errors and adjusts them proportionally. Real brains don't have backprop (this is an open neuroscience debate), but the effect is similar — repeated exposure to language patterns gradually strengthens the right connections.

### Phase 6: The Full Training Loop

```
 11,000 TRAINING STEPS — A Lifetime of Language Exposure
 ════════════════════════════════════════════════════════

 Step 1:     Brain is random noise. "The" → predicts "xkf7"
             Loss: 9.2 (catastrophically wrong)
             Every synapse adjusts dramatically.

 Step 100:   Brain learned common words. "The" → predicts "the"
             Loss: 5.1 (learning basic patterns)
             Grammar starting to emerge in attention layers.

 Step 1000:  Brain understands syntax. "The dog" → predicts "is"
             Loss: 3.4 (knows word order, basic facts)
             MLP layers storing "dog = animal" type knowledge.

 Step 3000:  Brain handles context. "The dog chased the cat and it"
             → predicts "was" (knows "it" = dog, the chaser)
             Loss: 2.3 (real comprehension emerging)

 Step 8000:  Brain near peak. Subtle distinctions:
             "The dog chased the cat and it was" → "tired" (not "happy")
             Loss: 2.1 (approaching the limits of 16.4M synapses)

 Step 11000: Training ends (10-min wall clock).
             Loss: 2.08 → val_bpb: 1.2077
             The brain has learned as much as it can in this lifetime.

 ┌──────────────────────────────────────────────────────────┐
 │  LEARNING CURVE (val_bpb over training)                  │
 │                                                          │
 │  4.0 ┤■                                                  │
 │      │ ■                                                 │
 │  3.0 ┤   ■                                               │
 │      │     ■■                                            │
 │  2.0 ┤        ■■■■■                                      │
 │      │              ■■■■■■■■■■■■■■■■■■                   │
 │  1.2 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─■■■■■ ← plateau │
 │      │                                                   │
 │  1.0 ┤  (leaderboard brains reach here)                  │
 │      └──┬────┬────┬────┬────┬────┬────┬────┬─            │
 │         0   1k   2k   4k   6k   8k  10k  11k            │
 │                      steps                               │
 └──────────────────────────────────────────────────────────┘
```

**Brain analogy:** This is a compressed version of human language acquisition. A child hears ~30,000 words/day and takes ~4 years to become fluent (~44 million words). Our model processes ~8 billion tokens in 11,000 steps — equivalent to ~700 years of human language exposure crammed into 10 minutes. The plateau at step ~8000 is like an adult's language ability — still improving, but the easy gains are exhausted. The remaining gap to the leaderboard (1.2077 vs 1.0856) is the difference between a competent speaker and a poet.

## Two Brains: Ours vs. the Leaderboard

A neuroscientist looking at our model and Kevin Clark's (rank 5) would see two very different brains.

### Our Brain — Run 11 (dim=448, 2× MLP, 9L, SP8192)

```
                    ┌─────────────────────────────┐
                   ╱  CEREBRAL CORTEX (MLP 44.2%)  ╲
                  ╱   Long-term knowledge store      ╲
                 ╱    "Paris is the capital of..."     ╲
                ╱      896 neurons wide per layer        ╲
               ╱        7.2M synapses total               ╲
              │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
              │                                           │
              │    PREFRONTAL CORTEX (Attention 33.3%)    │
              │    Working memory / focus / reasoning     │
              │    "Which noun does 'it' refer to?"       │
              │    5.5M synapses, 4 KV focus channels     │
              │                                           │
              │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
              │                                           │
              │     WERNICKE'S AREA (Embedding 22.4%)     │
              │     Dictionary: 8,192 words known         │
              │     3.7M synapses (word ↔ meaning)        │
              │                                           │
              └───────────────────────────────────────────┘

              Total brain mass: ~16.4M synapses
              Cortex thickness: 896 neurons (2× dim)
              Processing depth: 9 layers of thought
              Compression: int8+zlib → 15.35 MB
```

**Diagnosis:** A small, efficient brain. The dictionary (Wernicke's) is oversized at 22% — this brain knows 8,192 words but has a thin cortex to reason about them. Like a person with a huge vocabulary but limited life experience. The cortex is only 896 neurons wide — it can store facts but runs out of room quickly.

### Leaderboard Brain — Kevin Clark Style (dim=512, 4× MLP, 11L+recurrence, SP4096)

```
          ┌─────────────────────────────────────────────────────┐
         ╱          CEREBRAL CORTEX (MLP 68.2%)                  ╲
        ╱           Long-term knowledge store                      ╲
       ╱            "Paris is the capital of..."                     ╲
      ╱             "The Eiffel Tower was built in 1889"              ╲
     ╱              "French GDP is $2.78 trillion"                     ╲
    ╱                2,048 neurons wide per layer                       ╲
   ╱                  23.1M synapses total                               ╲
  │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
  │                                                                      │
  │              PREFRONTAL CORTEX (Attention 25.6%)                     │
  │              Working memory / focus / reasoning                      │
  │              8.7M synapses, 4 KV focus channels                     │
  │                                                                      │
  │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
  │              WERNICKE'S AREA (Embedding 6.2%)                        │
  │              Dictionary: 4,096 words known                           │
  │              2.1M synapses                                           │
  └──────────────────────────────────────────────────────────────────────┘

              ┌──────────────────────────┐
              │   DEPTH RECURRENCE       │
              │   Layers 4-5 loop twice  │
              │   = "sleeping on it"     │
              │   13 virtual layers of   │
              │   thought from 11 unique │
              │   Same neurons, re-used  │
              └──────────────────────────┘

              Total brain mass: ~33.8M synapses
              Cortex thickness: 2,048 neurons (4× dim)
              Processing depth: 13 virtual layers (11 unique)
              Compression: GPTQ int6+brotli → ~16 MB
```

**Diagnosis:** A large brain with a dominant cortex — like a primate vs. a reptile. The cortex is 2.3× thicker (2,048 vs 896 neurons), storing far more knowledge per layer. The dictionary is smaller (4,096 words) but that's a deliberate tradeoff: fewer words, but deeper understanding of each one. Depth recurrence is like **sleeping on a problem** — the same neural circuits in layers 4-5 fire twice, refining mid-level representations without growing the brain.

### Side-by-Side Comparison

```
         OUR BRAIN (16.4M)              LEADERBOARD BRAIN (33.8M)

      ┌───────────────────┐        ┌─────────────────────────────┐
      │░░░░░░░░░░░░░░░░░░░│        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
      │░░ CORTEX  44.2% ░░│        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
      │░░ 896 wide    ░░░░│        │▓▓▓ CORTEX  68.2%  ▓▓▓▓▓▓▓▓▓│
      │░░ 7.2M params ░░░░│        │▓▓▓ 2,048 wide     ▓▓▓▓▓▓▓▓▓│
      │░░░░░░░░░░░░░░░░░░░│        │▓▓▓ 23.1M params   ▓▓▓▓▓▓▓▓▓│
      ├───────────────────┤        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
      │                   │        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
      │ ATTENTION  33.3%  │        ├─────────────────────────────┤
      │ 5.5M params       │        │                             │
      │                   │        │  ATTENTION  25.6%           │
      ├───────────────────┤        │  8.7M params                │
      │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│        │                             │
      │▒ DICTIONARY 22.4%▒│        ├─────────────────────────────┤
      │▒ 8,192 words   ▒▒▒│        │ DICTIONARY  6.2%            │
      │▒ 3.7M params  ▒▒▒▒│        │ 4,096 words, 2.1M params   │
      └───────────────────┘        └─────────────────────────────┘

      Compressed: 15.35 MB          Compressed: ~16 MB (GPTQ)
      val_bpb: 1.2077               val_bpb: 1.0856
```

### The Neuroscience Interpretation

**Our model is like a small animal brain** — large language center relative to cortex. It knows many words (8,192) but has thin cortical layers to process them. Each "thought" passes through 896-neuron-wide layers, limiting how much knowledge can be stored and retrieved per inference.

**The leaderboard model is like a primate brain** — massive cortex dominates. It knows fewer words (4,096) but understands each one far more deeply. The 2,048-neuron-wide cortex can store 3.2× more factual associations per layer. And depth recurrence gives it a form of **iterative refinement** — like the brain replaying a thought during sleep to consolidate understanding.

The key insight: **in real brains, cortex size predicts intelligence far better than vocabulary size.** Crows (small brain, huge cortex ratio) outperform parrots (larger brain, smaller cortex ratio) on reasoning tasks. Similarly, the leaderboard proves that **a thicker cortex (4× MLP) with fewer words (SP4096) beats a thin cortex (2× MLP) with more words (SP8192)** — at least when you can compress the larger brain to fit.

The compression (GPTQ) is like **myelin sheathing** — it doesn't change the number of neurons, but it makes the signal transmission more efficient, allowing a larger brain to fit in the same skull (16 MB). Our brain uses crude compression (int8) that preserves signal quality but limits brain size. Their brain uses precise compression (GPTQ + quantization-aware training) that allows a 2× larger brain with only 0.012 BPB signal loss.

### Numbers at a Glance

| Metric | Our Brain | Leaderboard Brain | Analogy |
|--------|-----------|-------------------|---------|
| Cortex width | 896 | 2,048 | Mouse vs. macaque neocortex thickness |
| Cortex % | 44.2% | 68.2% | Reptile vs. primate cortex ratio |
| Dictionary | 8,192 words | 4,096 words | Polyglot vs. deep native speaker |
| Thought depth | 9 layers | 13 virtual (11 unique) | Quick glance vs. deliberate reflection |
| Depth recurrence | None | Layers 4-5 loop | — vs. sleep consolidation |
| Compression | int8 (crude) | GPTQ+QAT (precise) | Raw nerve vs. myelinated axon |
| Brain mass | 16.4M | 33.8M | 2× larger |
| Skull size | 15.35 MB | ~16 MB | Same skull |
| BPB | 1.2077 | 1.0856 | — |

## Cross-Reference

- `16_pg-leaderboard-techniques.md` — technique deep dives (SP8192, depth recurrence, etc.)
- `17_pg-leaderboard-annotated.md` — all leaderboard entries with components tagged
- `parameter-golf/train_gpt.py`:
  - Hyperparameters: line 43
  - CausalSelfAttention (gated Q projection, GQA): line 1214
  - MLP (LeakyReLU²): line 1314
  - Block (U-Net skip connections): line 1340
  - GPT class: line 1370
  - `_run_backbone` (forward pass): line 1430
  - `forward` (loss computation + SLM): line 1455
