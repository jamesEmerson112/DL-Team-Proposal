# Parameter Golf Leaderboard — Annotated (All 30 Entries)

**Author:** James Vo
**Date:** 2026-04-22
**Purpose:** Every leaderboard entry with every abbreviation decoded. For deep dives on the top 8 techniques, see note 16.

---

## Component Categories

Each technique is tagged with one of these labels:

| Tag | Meaning | Example |
|-----|---------|---------|
| **(tokenizer)** | Vocabulary / encoding changes | SP8192, SentencePiece BPE |
| **(architecture)** | Model structure — layers, attention, MLP, skip connections | Depth Recurrence, Parallel Residuals, XSA |
| **(optimizer)** | How weights get updated during training | Muon, Adam, MuonEq-R, EMA, SWA |
| **(quantization)** | Post-training compression to fit 16 MB | GPTQ, int6, int8, QAT, SDClip |
| **(eval)** | How the model is scored at validation time | Sliding Window, TTT |
| **(hyperparameter)** | Tuning knobs — LR, WD, warmdown, batch size | WD=0.095, warmdown=0.72 |
| **(activation)** | Nonlinearity function choice | ReLU², LeakyReLU² |
| **(init)** | Weight initialization strategy | Orthogonal init, Overtone SVD |
| **(compression)** | File-level compression of the artifact | zlib, zstd-22, Brotli-11, LZMA |

---

## Master Table — All Entries (Best to Worst)

### Rank 1 — 1.0810 BPB (Apr 09) — CURRENT SOTA
**SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT** — bigbag

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP8192 | tokenizer | 8192-entry dictionary — encodes words as whole pieces, not fragments |
| 3-Layer Depth Recurrence (layers 3,4,5) | architecture | Loops middle layers 3× for 17 virtual layers from 11 physical — free depth |
| Parallel Residuals (layer 7+) | architecture | Attention and MLP read from separate copies — lets each specialize |
| QK-Gain 5.25 | architecture | Learnable attention sharpness per head — tuned from 4.0 to 5.25 |
| MuonEq-R | optimizer | Row-normalized Muon — equalizes how fast each neuron learns |
| Score-First TTT (3 epochs, lr=0.005) | eval | Adapts weights on validation data, but only after scoring each chunk |
| GPTQ int6 matrices + int8 embeddings | quantization | Hessian-aware compression — int6 for compute layers, int8 for dictionary |
| SDClip (k=12.85 matrices, k=20.0 embeds) | quantization | Clips weight outliers at k × row_std before quantizing |
| EMA decay 0.9965 | optimizer | Smooth weight averaging across all of training |
| WD=0.095, MLR=0.022, warmdown=0.72 | hyperparameter | High regularization, long cooldown, tuned peak learning rate |
| Brotli-11 + LZMA wrapper | compression | Two-stage file compression — ~43% size reduction |

---

### Rank 2 — 1.0822 BPP (Apr 08)
**SP8192 + Parallel Residuals + Score-First TTT** — aryanbhosale

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP8192 | tokenizer | 8192-entry dictionary |
| Depth Recurrence (loop layers 4-5) | architecture | 2-layer loop for extra depth |
| Parallel Residuals (layer 7+) | architecture | Separate attention/MLP lanes with learned merge scalar (init 0.5) |
| QK-Gain 5.0 | architecture | Learnable attention sharpness |
| MuonEq-R | optimizer | Row-normalized Muon optimizer |
| Score-First TTT (3 epochs, lr=0.005) | eval | Legal test-time training — score before update |
| GPTQ int6 + int8 embeds + SDClip | quantization | Hessian-aware compression |
| Brotli compression | compression | File-level compression |

---

### Rank 3 — 1.0828 BPB (Apr 06)
**SP8192 + QK-Gain 5 + Legal Score-First TTT** — dexhunter

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP8192 | tokenizer | 8192-entry dictionary |
| QK-Gain 5.0 (up from 4.0) | architecture | Single-knob improvement — raised attention scaling |
| Depth Recurrence (loop layers 4-5, at 50%) | architecture | Activates mid-training to avoid early overhead |
| Score-First TTT | eval | Legal TTT — score under no_grad, then train on scored tokens |
| Full stack from PR #1394 unchanged | — | Same base as rank 5 (MuonEq-R, GPTQ, SDClip, etc.) |

---

### Rank 4 — 1.0835 BPB (Apr 06)
**SP8192 + Parallel Residuals + Hessian-Aware SDClip** — Robby Sneiderman

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP8192 | tokenizer | 8192-entry dictionary |
| Parallel Residuals (layer 7+) | architecture | Separate attention/MLP residual lanes |
| Hessian-Aware SDClip (lambda=0.175) | quantization | Weights importance from Hessian modulates clip threshold |
| Progressive Recurrence (50%->65%) | architecture | Staged depth recurrence — gradual activation |
| GPTQ int6 + int8 embeds | quantization | Standard compression stack |

---

### Rank 5 — 1.0856 BPB (Apr 05)
**SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip** — Kevin Clark

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP8192 | tokenizer | 8192-entry dictionary (jumped from 4096) |
| GPTQ Embeddings (int8) | quantization | Quantized the dictionary itself — enabled large vocab to fit |
| Depth Recurrence (layers 4-5, 2× loop) | architecture | Loop middle layers for free depth |
| MuonEq-R | optimizer | Row-normalized Muon |
| SDClip (k=12.85) | quantization | Principled entropy-driven clipping vs brute-force search |
| Removed value embeddings | architecture | Simplified — removed BigramHash and SmearGate |
| ShuffledSequenceLoader | hyperparameter | Better data ordering during training |

> **Nanochat:** BigramHash was adopted at d12 (our scale) then reverted at d25. SmearGate was negligible. Removal here aligns with nanochat's findings at larger scale.

---

### Rank 6 — 1.0897 BPB (Apr 04)
**SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R** — aryanbhosale

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP4096 | tokenizer | 4096-entry dictionary — 4× baseline |
| Depth Recurrence (layers 4,5) | architecture | 13 virtual layers from 11 physical — zero extra params |
| Parallel Residuals (layer 7+) | architecture | Separate attention/MLP lanes |
| MuonEq-R | optimizer | Row-normalized Muon |
| QK-Gain 5.0 | architecture | Learnable attention sharpness |
| MLP 4× | architecture | Wider feedforward network (up from 2×) |
| GPTQ int6 + Brotli | quantization | Full compression pipeline |

---

### Rank 7 — 1.0912 BPB (Apr 03)
**MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ** — dexhunter

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| MuonEq-R | optimizer | Row-normalized Muon — first appearance as a named technique |
| Depth Recurrence (layers 4,5) | architecture | Loop middle layers |
| WD=0.090 | hyperparameter | Higher weight decay — shrinks weights for better compression |
| All-Int6 GPTQ (all 66 layers) | quantization | Every layer at int6 — WD made weights small enough |
| SP4096 + MLP 4× | tokenizer / architecture | Same base as rank 8 |
| Brotli-11 | compression | File compression |

Key insight: **WD and quantization are synergistic** — higher WD shrinks weights, making int6 more accurate.

---

### Rank 8 — 1.0979 BPB (Apr 01)
**4096-Vocab + Larger Model + High WD + Simplifications** — Kevin Clark

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SP4096 | tokenizer | 4096-entry dictionary — first vocab jump from 1024 |
| MLP 4× expansion | architecture | Wider feedforward (1024→2048 hidden) — more compute per layer |
| WD=0.085 | hyperparameter | Higher weight decay than baseline's 0.04 |
| Hessian GPTQ | quantization | Full-Hessian post-training quantization |
| XSA all 11 layers | architecture | Exclusive Self Attention on every layer |
| BigramHash embeddings | architecture | Extra embedding for common character pairs |
| Sigmoid-gated skip connections | architecture | Improved U-Net skip gates |
| Removed: TTT, SmearGate, value residuals | — | Simplified by removing techniques that didn't help at this scale |
| Brotli-11 | compression | File compression |

> **Nanochat:** BigramHash adopted at d12, reverted at d25. SmearGate negligible. XSA had better step quality but not wall clock — for PG (BPB metric), worth testing.

Key insight: **RMS weight magnitude correlates with compressibility (R²=0.99)**

---

### Rank 9 — 1.1063 BPB (Mar 31)
**Parallel Residuals + Mini Depth Recurrence** — Marko Sisovic

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Parallel Residuals (layer 7+) | architecture | First appearance — attention/MLP on separate lanes |
| Mini Depth Recurrence (layers 4,5) | architecture | First appearance — loop 2 middle layers, activate at step 3000 |
| Untied repeated MLPs | architecture | Recurred layers share attention but get separate MLP weights |
| AR Self-Gen GPTQ | quantization | Calibrate quantization on model-generated text, not training data |
| Mixed int6/int8 | quantization | int6 for most layers, int8 for sensitive ones |
| 11 layers | architecture | Up from baseline's 9 |
| Disable layer-0 attention | architecture | Skip attention in first layer — it doesn't help |

---

### Rank 10 — 1.1147 BPB (Mar 25) — First Merged SOTA
**11L AR Self-Gen GPTQ + XSA** — abaybektursun

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Full-Hessian GPTQ | quantization | Calibrate with full Hessian matrix — better than GPTQ-lite |
| AR Self-Generated calibration | quantization | Model generates its own calibration data — no external data needed |
| XSA all 11 layers | architecture | Exclusive Self Attention everywhere — richer context mixing |
| BigramHash (3072×112) | architecture | Large bigram embedding table |
| Selective ±1 pruning | quantization | Prune weights that are exactly ±1 after quantization |
| Parallel Muon | optimizer | Muon variant from PR #549 |
| LeakyReLU(0.5)² | activation | Preserves negative gradients |
| LZMA preset=9 | compression | Better compression than zlib for this weight distribution |

> **Nanochat:** LeakyReLU² had better per-step quality but slower wall clock (Mar 24). For PG (BPB metric, not speed), this favors testing it. BigramHash adopted at d12 (PG's scale).

---

### Rank 11 — 1.1194 BPB (Mar 23)
**LeakyReLU² + Legal Score-First TTT + Parallel Muon** — abaybektursun

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| LeakyReLU(0.5)² | activation | First appearance — keeps small negative gradients flowing |
| Score-First TTT | eval | First legal TTT — score chunk, then train on it, never cheat |
| Parallel Muon | optimizer | Muon variant with parallelized Newton-Schulz steps |
| Parameter Banking | optimizer | Technique for distributed optimizer state |

> **Nanochat:** LeakyReLU² better per-step, slower wall clock. PG cares about BPB — worth testing.

---

### Rank 12 — 1.1228 BPB (Mar 22)
**11L EMA + GPTQ-lite + warmdown3500** — signalrush

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| GPTQ-lite | quantization | Lightweight GPTQ — searches optimal clip percentile per row |
| EMA weight averaging | optimizer | Smooth continuous averaging (replaced SWA snapshots) |
| warmdown=3500 | hyperparameter | Extended learning rate cooldown — 3500 steps |
| Late QAT at threshold 0.15 | quantization | Quantization-aware training kicks in late |
| Tight SWA | optimizer | Supplementary weight averaging |

---

### Rank 13 — 1.1248 BPB (Mar 21)
**11L Partial RoPE + LN Scale + EMA + XSA4** — jfprincz

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Partial RoPE (16/64 dims) | architecture | Only rotate 16 of 64 head dims — rest are position-invariant |
| LayerNorm Scale (1/sqrt(layer+1)) | architecture | Deeper layers get smaller scale — stabilizes training |
| EMA | optimizer | Exponential moving average of weights |
| XSA on last 4 layers | architecture | Exclusive Self Attention only on deepest layers |
| Late QAT | quantization | (Actually non-functional due to torch.compile constant folding) |

> **Nanochat:** Partial RoPE slightly worse (Mar 24). LN Scale did not help (Mar 24). Both confirmed negative.

---

### Rank 14 — 1.1271 BPB (Mar 20)
**11L XSA4 + EMA + Int6 MLP3x** — jfprincz

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| XSA on last 4 layers | architecture | Exclusive Self Attention — extra attention computation |
| EMA (decay=0.997) | optimizer | First appearance of EMA — replaced SWA for smoother averaging |
| Int6 mixed quantization | quantization | Int6 for blocks, int8 for embeddings |
| MLP 3× | architecture | Wider feedforward (up from 2×) |
| 11 layers | architecture | Deeper model funded by int6 compression savings |

---

### Rank 15 — 1.1307 BPB (Mar 20)
**11L Efficient Partial XSA** — unnir

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Efficient Partial XSA (last 3 layers) | architecture | First XSA appearance — GQA-aware reshape, ~2ms/step overhead |
| Flash Attention 3 | architecture | Faster attention kernel — Hopper GPU optimized |
| SWA every 120 steps | optimizer | Stochastic Weight Averaging — periodic snapshots |
| 11 layers | architecture | Deeper model |

> **Nanochat:** FA3 gives ~9% tok/sec improvement (Jan 11). Requires Hopper GPU (sm90+).

---

### Rank 16 — 1.1428 BPB (Mar 20)
**10L Int5-MLP + BigramHash(10240)** — thwu1

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Int5 MLP quantization | quantization | 5-bit precision for MLP weights — more aggressive than int6 |
| Int6 attention | quantization | 6-bit for attention weights |
| BigramHash (10240 buckets) | architecture | Large bigram hash table for character-pair patterns |
| SWA (start_frac=0.4) | optimizer | Weight averaging starting at 40% of training |
| 10 layers | architecture | One extra layer funded by int5 savings |
| U-Net skip connections | architecture | Encoder-decoder style residual shortcuts |

> **Nanochat:** BigramHash adopted at d12 (PG's scale), reverted at d25. Supports using it at our model size.

---

### Rank 17 — 1.1458 BPB (Mar 20)
**Int6 MLP3x + SmearGate + BigramHash** — Raahil Shah

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Int6 per-row quantization | quantization | 6-bit weights with per-row scaling |
| MLP 3× expansion | architecture | Wider feedforward — 1024→1536 hidden |
| SmearGate | architecture | Learnable gate that blends bigram info into token embeddings |
| BigramHash (4096 buckets) | architecture | Hash-based bigram embeddings |
| Orthogonal init | init | Initialize weights as orthogonal matrices — better starting point |
| Muon WD=0.04 | optimizer | Muon with weight decay |
| SWA every 50 steps | optimizer | Frequent weight averaging snapshots |
| zstd-22 | compression | Better compression than zlib for int6 data |

> **Nanochat:** SmearGate negligible (Mar 24). Orthogonal init did not help (Mar 24). BigramHash adopted at d12, reverted at d25.

---

### Rank 18 — 1.1502 BPB (Mar 20)
**11L MLP3x + Int6 QAT** — aruniyer

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Int6 QAT via STE | quantization | Train with simulated 6-bit precision — Straight-Through Estimator |
| MLP 3× | architecture | Wider feedforward |
| 11 layers | architecture | Deeper model — funded by int6 compression |
| Sliding window eval (stride=64) | eval | Score with overlapping windows for more context |
| Muon WD=0.04 | optimizer | Muon with weight decay |
| zstd-22 | compression | File compression |

---

### Rank 19 — 1.1556 BPB (Mar 19)
**SmearGate + OrthoInit + Muon WD** — aquariouseworkman

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| SmearGate | architecture | Bigram blending layer — learns to mix character-pair info |
| Orthogonal init | init | Better weight initialization |
| BigramHash | architecture | Hash-based bigram embeddings |
| MLP 3× | architecture | Wider feedforward |
| Int6 STE QAT | quantization | Quantization-aware training at 6-bit |
| Sliding window eval | eval | Overlapping window evaluation |
| Muon WD=0.01 | optimizer | Light weight decay on Muon optimizer |

> **Nanochat:** SmearGate negligible. Orthogonal init did not help. Both confirmed negative.

---

### Rank 20 — 1.1570 BPB (Mar 24)
**Ternary Quantization** — Ciprian-Florin Ifrim

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| BitNet b1.58 ternary {-1, 0, +1} | quantization | Extreme: weights are only -1, 0, or +1 — 73.7M params |
| 768d / 10L | architecture | Width-over-depth strategy — wider but fewer layers |
| FP8 QAT | quantization | 8-bit floating point quantization-aware training |
| 8192 BPE | tokenizer | Large vocab (unusual for this date — Mar 24) |
| YaRN | architecture | Extended context via rotary position scaling |
| NeoMuon (3 Newton-Schulz steps) | optimizer | Faster Muon variant with fewer orthogonalization steps |
| Base-3 LZMA | compression | Custom compression for ternary weights |

> **Nanochat:** FP8 training gave 17% speed gain but only 5% capability-matched at d24 (Feb 2). At small scale, overhead may dominate.

Outlier entry — unique approach but didn't beat contemporaries.

---

### Rank 21 — 1.1586 BPB (Mar 19)
**10L Int6 QAT + Zstd MLP2.6x** — yahya010

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Int6 QAT + STE | quantization | Quantization-aware training at 6-bit |
| MLP 2.6× (hidden=1344) | architecture | Slightly wider feedforward |
| Muon momentum=0.99 | optimizer | Higher momentum for smoother updates |
| Seq_len=2048 | hyperparameter | Longer training sequences |
| FP16 embedding | quantization | Keep dictionary in half-precision (not int8) |
| zstd-22 | compression | File compression |
| Sliding window eval | eval | Overlapping evaluation windows |

---

### Rank 22 — 1.1630 BPB (Mar 19)
**Mixed Quant + Sliding Window Eval** — aquariouseworkman

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Mixed int6/int8 quantization | quantization | Int6 for blocks, int8 for embeddings — reduces quant penalty from +0.048 to +0.0015 BPB |
| MLP 3× | architecture | Wider feedforward |
| Sliding window eval (stride=64) | eval | Overlapping windows for better context |
| STE QAT | quantization | Quantization-aware training |

---

### Rank 23 — 1.1748 BPB (Mar 19)
**Muon WD + 10 layer** — notapplica

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| 10 layers | architecture | One extra layer over baseline's 9 |
| Muon WD=0.02 | optimizer | Weight decay on Muon — pushes unused weights toward zero |
| Sliding window eval (stride=64) | eval | Better context per scored token |
| FP16 tied embedding | quantization | Keep dictionary in half-precision |
| Overtone SVD init | init | Spectral SVD initialization — better starting weights |
| Phase-transition residual mixing | architecture | Learned blending of residual streams |

---

### Rank 24 — 1.1925 BPB (Mar 19)
**Sliding Window Eval** — Matthew Li

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Sliding window eval (stride=64) | eval | First appearance — score with 960+ tokens of context instead of averaging 512 |

Single change, big win. Every subsequent entry adopted this.

---

### Rank 25 — 1.1928 BPB (Mar 19)
**LoRA TTT** — samacqua

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| LoRA Test-Time Training | eval | Attach small adapter layers, train them on validation chunks |
| Document-aware conditioning | eval | Isolate adaptation per document — don't bleed context |
| Sliding window chunks | eval | Process validation in chunks |

Early TTT approach — later replaced by score-first TTT for legality.

---

### Rank 26 — 1.2014 BPB (Mar 19)
**4k seq length** — Spokane Way

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Seq_len=4096 | hyperparameter | 4× longer training sequences — more context per step |
| Muon momentum=0.99 | optimizer | Higher momentum |
| Lower learning rates | hyperparameter | Gentler updates for stability with longer sequences |
| Extended warmup/warmdown | hyperparameter | Longer schedule ramps |

---

### Rank 27 — 1.206 BPB (Mar 18)
**2048 seq length** — Spokane Way

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| Seq_len=2048 | hyperparameter | 2× longer sequences — first context length increase |
| Tuned learning rates | hyperparameter | Adjusted for longer sequences |

---

### Rank 28 — 1.2147 BPB (Mar 18)
**int6 mixed precision** — Nan Liu

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| 10 layers | architecture | One extra layer |
| Mixed int8/int6 (middle layers int6) | quantization | First int6 experiment — selective on layers 3-6 |
| Lower learning rates (0.02/0.03) | hyperparameter | Gentler updates |

---

### Rank 29 — 1.2197 BPB (Mar 18)
**fp16 Embed** — Renier Velazco

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| FP16 embedding passthrough | quantization | Keep dictionary in half-precision instead of int8 — cuts quant loss from 0.007 to 0.0005 BPB |
| MLP hidden 992 (down from 1024) | architecture | Slightly smaller MLP to compensate for fp16 size |
| Warmdown=3600 | hyperparameter | Extended cooldown — default assumed more steps than 10 min allows |
| Matrix LR=0.06 | hyperparameter | Higher peak learning rate |

---

### Rank 30 — 1.2244 BPB (Mar 17)
**Naive Baseline** — OpenAI

| Technique | Component | Plain English |
|-----------|-----------|---------------|
| 9 layers, 512 dim | architecture | Base model shape |
| SP1024 | tokenizer | 1024-entry dictionary |
| GQA (8 heads, 4 KV heads) | architecture | Grouped Query Attention |
| Muon + Adam | optimizer | Muon for matrices, Adam for embeddings |
| ReLU² | activation | Squared ReLU activation |
| RoPE | architecture | Rotary Position Embedding |
| RMSNorm | architecture | Root Mean Square normalization |
| U-Net skip connections | architecture | Encoder-decoder residual shortcuts |
| Tied embeddings | architecture | Input embedding = output head (saves params) |
| Int8 + zlib | quantization | Basic post-training compression |

---

## Technique Glossary

Every unique abbreviation across all 30 entries:

| Abbreviation | Full Name | Component | Plain English |
|---|---|---|---|
| **SP1024/4096/8192** | SentencePiece BPE | tokenizer | Dictionary size — how many subword pieces the model knows |
| **GQA** | Grouped Query Attention | architecture | Multiple query heads share fewer key/value heads — saves params |
| **RoPE** | Rotary Position Embedding | architecture | Encodes token position by rotating Q/K vectors |
| **Partial RoPE** | Partial Rotary Embedding | architecture | Only rotate some head dims — rest are position-invariant |
| **RMSNorm** | Root Mean Square Normalization | architecture | Normalize by RMS of activations — simpler than LayerNorm |
| **LN Scale** | LayerNorm Scale | architecture | Scale factor 1/sqrt(layer+1) — deeper layers get smaller updates |
| **ReLU²** | Squared ReLU | activation | relu(x)² — zeroes negatives, squares positives |
| **LeakyReLU²** | Leaky ReLU Squared | activation | leaky_relu(x, 0.5)² — preserves small negative gradients |
| **XSA** | Exclusive Self Attention | architecture | Extra attention computation for richer context mixing |
| **DR** | Depth Recurrence | architecture | Run same layers multiple times — free depth, zero extra params |
| **ParResid** | Parallel Residuals | architecture | Attention and MLP read from separate input copies (GPT-J style) |
| **BigramHash** | Bigram Hash Embeddings | architecture | Hash table mapping character pairs to extra embedding dims |
| **SmearGate** | Bigram Blending Gate | architecture | Learnable gate that blends bigram info into token embeddings |
| **U-Net Skips** | U-Net Skip Connections | architecture | Encoder layers store state, decoder layers retrieve in reverse |
| **QK-Gain** | Query-Key Gain | architecture | Learnable per-head scaling for Q·K dot products |
| **Muon** | Muon Optimizer | optimizer | Orthogonalizes gradient matrices via Newton-Schulz before applying |
| **MuonEq-R** | Row-Normalized Muon | optimizer | Muon with per-row gradient normalization — equalizes neuron updates |
| **NeoMuon** | Neo Muon | optimizer | Faster Muon variant — fewer Newton-Schulz steps |
| **Adam / AdamW** | Adam Optimizer | optimizer | Adaptive learning rate per weight based on gradient history |
| **EMA** | Exponential Moving Average | optimizer | Continuously smooth weight average across training |
| **SWA** | Stochastic Weight Averaging | optimizer | Periodic weight snapshots averaged together |
| **WD** | Weight Decay | hyperparameter | Pushes unused weights toward zero — acts as regularizer |
| **MLR** | Matrix Learning Rate | hyperparameter | Peak learning rate for Muon-optimized matrix weights |
| **Warmdown** | Learning Rate Warmdown | hyperparameter | Linear LR decay to zero over final N% of training |
| **QAT** | Quantization-Aware Training | quantization | Simulate low-precision during training to reduce compression loss |
| **STE** | Straight-Through Estimator | quantization | Trick for backprop through discrete quantization — pretend rounding is identity |
| **GPTQ** | GPTQ Quantization | quantization | Post-training Hessian-aware weight compression |
| **GPTQ-lite** | Lightweight GPTQ | quantization | Simpler GPTQ — per-row optimal clip percentile search |
| **SDClip** | Std-Dev Clipping | quantization | Clip outliers at k × row_std before quantizing |
| **AR Self-Gen** | Auto-Regressive Self-Generated | quantization | Model generates its own calibration data for GPTQ |
| **Int5 / Int6 / Int8** | Integer Quantization | quantization | Reduce weight precision to 5/6/8-bit integers |
| **FP16** | Half-Precision Float | quantization | 16-bit floating point — more precise than int8, costs 2× space |
| **Ternary** | Ternary Quantization | quantization | Extreme: weights are only {-1, 0, +1} |
| **FP8** | 8-bit Floating Point | quantization | 8-bit float format — used in QAT on Hopper GPUs |
| **TTT** | Test-Time Training | eval | Adapt model weights on already-scored validation tokens |
| **LoRA TTT** | LoRA Test-Time Training | eval | Attach small adapter layers, train only those at eval time |
| **Score-First TTT** | Score-First Test-Time Training | eval | Score chunk first (no_grad), then train — ensures legality |
| **Sliding Window** | Sliding Window Evaluation | eval | Overlapping windows (stride=64) — every token scored with 960+ context |
| **YaRN** | Yet Another RoPE Extension | architecture | Extends RoPE to handle longer sequences |
| **OrthoInit** | Orthogonal Initialization | init | Initialize weight matrices as orthogonal — decorrelates neurons |
| **Overtone SVD** | Spectral SVD Initialization | init | Spectral decomposition-based weight initialization |
| **zlib** | zlib Compression | compression | Standard deflate compression |
| **zstd-22** | Zstandard Level 22 | compression | High-ratio compression — better than zlib for int6 data |
| **Brotli-11** | Brotli Level 11 | compression | Google's compression — best ratio for this weight distribution |
| **LZMA** | Lempel-Ziv-Markov-chain | compression | High-ratio compression used for self-extracting wrappers |

### Nanochat-Only Techniques (not on PG leaderboard)

Techniques tested in `nanochat/dev/LOG.md` that don't appear on the PG leaderboard but inform our experiment planning:

| Abbreviation | Full Name | Component | Plain English | Nanochat Result |
|---|---|---|---|---|
| **SwiGLU** | Switched Gated Linear Unit | activation | 3-projection MLP with SiLU gating — theoretically better but ReLU² wins empirically | NEGATIVE (d12, d24) |
| **MoE** | Mixture of Experts | architecture | 8 routed experts + 1 shared, top-2 routing — DeepSeekV3 style | NEGATIVE (overhead) |
| **MuonH / Hyperball** | Hypersphere-Constrained Muon | optimizer | Constrains weights to sphere of initial norm radius | NEGATIVE (all sweeps) |
| **MTP** | Multi-Token Prediction | eval | Predict next N tokens per position with weighted loss | NEGATIVE (+13 GB, worse) |
| **Value Embeddings (VE)** | Value Embeddings | architecture | Per-layer embedding tables added to V tensor — huge capacity at zero FLOP cost | POSITIVE (adopted) |
| **ClimbMix** | ClimbMix 400B Dataset | data | Curated 400B-token mix from NVIDIA — 27% training time reduction over FineWeb-EDU | POSITIVE (adopted) |

---

## Cross-Reference

- **Note 16** (`16_pg-leaderboard-techniques.md`) — Deep dives on the top 8 techniques with measured BPB gains
- **Note 13** (`13_pg-vs-nanochat-architecture.md`) — PG vs nanochat comparison — most techniques are shared
- **Note 14** (`14_modded-nanogpt-lineage.md`) — PG descends from modded-nanogpt

## Sources

- PG leaderboard data: `parameter-golf/records/track_10min_16mb/*/README.md` (33 folders) and `parameter-golf/README.md`
- Nanochat experiment results: `nanochat/dev/LOG.md` (Jan–Mar 2026, ~30 experiments)
