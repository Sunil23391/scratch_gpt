import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.kv_cache.cache import KVCache

class MultiHeadAttention(nn.Module):
    """Multi‑Head Attention with optional Grouped‑Query Attention (GQA).
    $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, gqa_groups: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.total_dim = n_heads * head_dim
        self.gqa_groups = gqa_groups
        # Projections
        self.q_proj = nn.Linear(dim, self.total_dim, bias=False)
        # For GQA, keys/values are shared across groups
        self.k_proj = nn.Linear(dim, (n_heads // gqa_groups) * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, (n_heads // gqa_groups) * head_dim, bias=False)
        self.out_proj = nn.Linear(self.total_dim, dim, bias=False)
        self.kv_cache = KVCache()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = False):
        # x: [batch, seq_len, dim]
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.n_heads // self.gqa_groups, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads // self.gqa_groups, self.head_dim)
        if use_cache:
            K, V = self.kv_cache.append(K, V)
            
        # For GQA, repeat KV heads to match number of query heads
        if self.gqa_groups > 1:
            # K shape: [B, S, n_kv_heads, D] -> [B, S, n_heads, D]
            K = torch.repeat_interleave(K, self.gqa_groups, dim=2)
            V = torch.repeat_interleave(V, self.gqa_groups, dim=2)
            
        # Compute scaled dot‑product attention
        scores = torch.einsum('bthd,bshd->bths', Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bths,bshd->bthd', attn, V)
        out = out.reshape(B, T, self.total_dim)
        return self.out_proj(out)
