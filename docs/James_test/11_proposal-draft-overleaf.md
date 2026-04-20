# Proposal Draft — Overleaf LaTeX (with inline citations)

> **Status:** Draft v2 — Overleaf-ready with `\cite{}` references
> **Reuses:** existing refs [1] nanochat, [2] FineWeb, [3] Parameter Golf
> **Adds:** 4 new bibliography entries (gated attention, DCLM, RL limits, superposition)

---

## LaTeX — Section text (copy into Overleaf)

```latex
\section{Approach}

\subsection{Preparation}
Firstly, we will fork nanochat \cite{nanochat} to study the project, then adapt it to operate under OpenAI's Parameter Golf \cite{parametergolf} constraints (1,024-token vocabulary, 16 MB artifact limit, 10-minute training budget on 8xH100/H200 GPUs).

\subsection{Phase 1}
We will modify nanochat's tokenizer module to load the competition's SentencePiece model, update the data loader in \texttt{base\_train.py} to read the competition's shard format, and configure the model architecture to fit within 16 MB. Then we will train an unmodified baseline model and log its learning curve. The model is pretrained on FineWeb \cite{fineweb}, a 15-trillion token curated web text dataset that has been shown to outperform other open pretraining corpora.

\subsection{Experiment}
With the baseline established, each member pursues a different improvement direction to improve the learning curves then synchronize with each other to improve the model further by doing the following:

\begin{itemize}
    \item \textbf{Architectural modifications} --- informed by recent work on gated attention \cite{gatedattn} (NeurIPS 2025 Best Paper), which inserts learnable gating scalars into attention layers to improve training stability and eliminate attention sinks.
    \item \textbf{Dataset variations} --- experimenting with data ordering and shard selection, informed by evidence that data curation outweighs data quantity \cite{dclm}.
    \item \textbf{Hyperparameter fine-tuning and Reinforcement Learning} --- testing whether GRPO-based RL improves BPB at small model scales, noting recent findings \cite{rllimits} that RL redistributes rather than creates reasoning capacity. Scaling behavior is guided by Chinchilla-optimal laws, whose theoretical basis in representation superposition has been recently formalized \cite{superposition}.
\end{itemize}

\subsection{Results}
Our models are evaluated using Bits Per Byte (BPB) on the FineWeb validation set \cite{parametergolf} --- a tokenizer-agnostic compression metric where lower scores indicate better language modeling, with the current competition baseline at 1.2244 BPB.
```

---

## LaTeX — New bibliography entries (append after existing [9])

```latex
\bibitem{gatedattn}
Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., \& Lin, J. (2025).
Gated Attention for Large Language Models.
\textit{NeurIPS 2025 Best Paper}.
arXiv:2505.06708.

\bibitem{rllimits}
Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., \& Huang, G. (2025).
Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
\textit{NeurIPS 2025}.
arXiv:2504.13837.

\bibitem{superposition}
Liu, Y., Liu, Z., \& Gore, J. (2025).
Superposition Yields Robust Neural Scaling.
\textit{NeurIPS 2025}.
arXiv:2505.10465.

\bibitem{dclm}
Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S., et al. (2024).
DataComp-LM: In Search of the Next Generation of Training Data.
\textit{NeurIPS 2024}.
arXiv:2406.11794.
```

---

## Citation mapping

| Cite key | What | Where cited |
|----------|------|-------------|
| `\cite{nanochat}` | nanochat repo (existing [1]) | 2.1 Preparation |
| `\cite{fineweb}` | FineWeb dataset (existing [2]) | 2.2 Phase 1 |
| `\cite{parametergolf}` | Parameter Golf (existing [3]) | 2.1 Preparation, 2.4 Results |
| `\cite{gatedattn}` | Gated Attention — NeurIPS 2025 Best Paper (NEW) | 2.3 Experiment |
| `\cite{dclm}` | DataComp-LM — NeurIPS 2024 (NEW) | 2.3 Experiment |
| `\cite{rllimits}` | RL Reasoning Limits — NeurIPS 2025 (NEW) | 2.3 Experiment |
| `\cite{superposition}` | Superposition Scaling — NeurIPS 2025 (NEW) | 2.3 Experiment |

---

## Note
Check that `\cite{nanochat}`, `\cite{fineweb}`, and `\cite{parametergolf}` match the cite keys your teammates used for those references. If they used different keys, update the `\cite{}` calls above to match.
