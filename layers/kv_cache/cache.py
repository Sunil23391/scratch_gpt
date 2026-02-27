import torch

class KVCache:
    """Simple static/rolling KV cache for autoregressive inference.
    Stores past keys and values and concatenates new ones.
    """
    def __init__(self):
        self.keys = []
        self.values = []

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new keys and values.
        Args:
            k: Tensor of shape [batch, seq_len, heads, head_dim]
            v: Tensor of shape [batch, seq_len, heads, head_dim]
        Returns:
            Concatenated keys and values along sequence dimension.
        """
        self.keys.append(k)
        self.values.append(v)
        K = torch.cat(self.keys, dim=1)  # concat on sequence dim
        V = torch.cat(self.values, dim=1)
        return K, V

    def reset(self):
        """Clear the cache (e.g., at start of new generation)."""
        self.keys.clear()
        self.values.clear()
