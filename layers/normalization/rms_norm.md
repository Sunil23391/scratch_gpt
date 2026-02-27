# RMSNorm Documentation

**Component:** RMSNorm (Root Mean Square Layer Normalization)

The RMSNorm operation normalizes the input tensor $x$ by its root-mean-square (RMS) value and then scales it with a learnable gain $g$:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot g
$$

- $x \in \mathbb{R}^{B \times S \times d}$ where $B$ is batch size, $S$ sequence length, $d$ hidden dimension.
- $\epsilon$ is a small constant for numerical stability (default $1\times10^{-6}$).
- $g \in \mathbb{R}^{d}$ is a learnable parameter initialized to ones.

**Shape Flow:**
- Input: `[batch, seq_len, dim]`
- Output: `[batch, seq_len, dim]`

**Reference Implementation:** See `layers/normalization/rms_norm.py`.
