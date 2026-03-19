# Key Terms & Glossary

## Training Pipeline

**Pretrain → SFT → RLHF** — the standard 3-step pipeline for building a chat model:

1. **Pretraining** — Learn language by predicting the next token on massive web text. The model learns grammar, facts, reasoning patterns, but doesn't know how to chat yet.

2. **SFT (Supervised Fine-Tuning)** — Teach the pretrained model to follow instructions and have conversations by training on curated examples (Q&A pairs, multi-turn conversations). Turns an autocomplete engine into a chatbot.

3. **RLHF (Reinforcement Learning from Human Feedback)** — Reward the model for producing good outputs. In nanochat, this is simplified to **GRPO** (Group Relative Policy Optimization) applied to math problems (GSM8K).

## Tokenization

**BPE (Byte Pair Encoding)** — Algorithm to convert text into tokens (numbers) the model can process:
1. Start with individual characters
2. Find the most frequent adjacent pair, merge into one token
3. Repeat until target vocabulary size is reached

Result: common words = 1 token, rare words = multiple tokens.
- nanochat: custom Rust-based BPE, 32K–65K vocab
- nanoGPT: reuses OpenAI's GPT-2 tokenizer (50K vocab) via tiktoken

## Metrics & Evaluation

**Learning Curve** — A plot showing how model performance changes over training. Specifically:
- **Training loss** and **validation loss** over epochs/steps
- **Training perplexity** and **validation perplexity** over epochs/steps
- Used to verify the model is converging and not overfitting
- **Required by TAs** as proof that training actually happened

> Note: The comparison table column "Ease of use" refers to *framework difficulty* (how hard it is to learn the tool). The ML meaning above — loss/perplexity plots — is what the TAs require in the final paper.

**Perplexity** — A measure of how "surprised" the model is by the data. Lower = better. Mathematically, it's `exp(loss)`. A perplexity of 5 means the model is roughly as uncertain as choosing between 5 equally likely options for the next token.

## Nanochat Pipeline Summary

| Step | What happens | Time |
|---|---|---|
| 1. Build tokenizer | Turn text into numbers (BPE) | ~30 min |
| 2. Pretrain | Learn language from web data (FineWeb-EDU) | ~2.5–3 hrs |
| 3. SFT | Learn to chat from example conversations | ~8–30 min |
| 4. RL (optional) | Get rewarded for correct math answers | ~1 hr |
