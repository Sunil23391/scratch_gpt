# SwiGLU Documentation

**Component:** SwiGLU (SwiGLU Activation)

The SwiGLU activation is embedded in an MLP block that uses three weight matrices:

$$
\text{MLP}(x) = (\text{SiLU}(xW_1) \otimes xW_3)W_2
$$

- $x \in \mathbb{R}^{B \times S \times d}$ is the input tensor.
- $W_1, W_3 \in \mathbb{R}^{d \times h}$ are up‑projection matrices.
- $W_2 \in \mathbb{R}^{h \times d}$ is the down‑projection matrix.
- $\text{SiLU}$ is the Sigmoid‑Linear Unit activation.
- The element‑wise product $\otimes$ is applied between the gated path and the up‑projection path.

**Shape Flow:**
- Input: `[batch, seq_len, dim]`
- Output: `[batch, seq_len, hidden_dim]`

**Reference Implementation:** See `layers/mlp/swiglu.py`.
