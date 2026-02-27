import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU activation and MLP block using three weight matrices.
    $$\text{MLP}(x) = (\text{SiLU}(xW_1) \otimes xW_3)W_2$$
    where $W_1, W_3$ are up-projections and $W_2$ is a down-projection.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        gate = F.silu(self.w1(x))          # [b, s, h]
        up = self.w3(x)                    # [b, s, h]
        out = self.w2(gate * up)           # [b, s, dim]
        return out
