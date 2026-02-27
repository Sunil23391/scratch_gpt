from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 32000
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    head_dim: int = 64  # dim // n_heads
    hidden_dim: int = 5504 # Standard SwiGLU expansion
    seq_len: int = 2048
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 10
    gqa_groups: int = 1  # set >1 for grouped‑query attention
    eps: float = 1e-6
    rope_base: float = 10000.0
