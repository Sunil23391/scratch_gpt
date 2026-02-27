# KV-Cache Documentation

**Component:** KV‑Cache (Key‑Value Cache)

The KV‑Cache stores previously computed key and value tensors during autoregressive generation, allowing the model to reuse past attention computations without recomputing them.

- `append(k, v)` adds new key/value tensors (shape `[batch, seq_len, heads, head_dim]`) to the cache and returns the concatenated tensors along the sequence dimension.
- `reset()` clears the cache, typically called at the start of a new generation sequence.

**Shape Flow:**
- Input to `append`: `k` and `v` of shape `[batch, seq_len, heads, head_dim]`
- Output from `append`: concatenated `K` and `V` of shape `[batch, total_seq_len, heads, head_dim]`

**Reference Implementation:** See `layers/kv_cache/cache.py`.
