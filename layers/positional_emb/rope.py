import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    """Rotary Positional Embedding using complex view.
    $$\mathbf{R}_{\theta} = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$$
    """
    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq * scale)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: [batch, seq_len, n_heads, head_dim]
        # inv_freq: [head_dim / 2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # freqs: [seq_len, head_dim / 2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # View x as complex
        # x_complex: [batch, seq_len, n_heads, head_dim / 2]
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        # Create rotation tensor in complex space
        # rot: [seq_len, head_dim / 2]
        # We need it as [1, seq_len, 1, head_dim / 2] for broadcasting
        rot = torch.polar(torch.ones_like(freqs), freqs)
        rot = rot[None, :, None, :]
        
        # Apply rotation
        x_rot = x_complex * rot
        
        # Return as real tensor
        return torch.view_as_real(x_rot).reshape_as(x)
