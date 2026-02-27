# Multi-Head Attention Documentation

**Component:** Multi‑Head Attention (MHA) with optional Grouped‑Query Attention (GQA)

The attention mechanism computes a weighted sum of value vectors based on the similarity of query and key vectors:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

- $Q, K, V \in \mathbb{R}^{B \times S \times H \times d_k}$ where $B$ is batch size, $S$ sequence length, $H$ number of heads, $d_k$ head dimension.
- For GQA, the number of query heads can differ from key/value heads; keys and values are shared across groups of queries.
- The implementation uses a `KVCache` to store past keys and values for efficient autoregressive generation.

**Shape Flow:**
- Input: `[batch, seq_len, dim]`
- Output: `[batch, seq_len, dim]`

**Reference Implementation:** See `layers/attention/attention.py`.
