# RoPE Documentation

**Component:** RoPE (Rotary Positional Embedding)

RoPE rotates query and key vectors in the complex plane based on their position in the sequence. For a given dimension $d$, the rotation matrix for position $t$ is:

$$
\mathbf{R}_{\theta_t} = \begin{bmatrix}\cos(\theta_t) & -\sin(\theta_t)\\ \sin(\theta_t) & \cos(\theta_t)\end{bmatrix},
\quad \theta_t = t \cdot \omega,
$$
where $\omega$ is a frequency vector derived from the inverse frequency buffer.

The implementation computes cosine and sine embeddings for each position, views the hidden states as complex numbers via `torch.view_as_complex`, and applies the rotation efficiently without extra memory allocations.

**Shape Flow:**
- Input: `[batch, seq_len, n_heads, head_dim]`
- Output: `[batch, seq_len, n_heads, head_dim]`

**Reference Implementation:** See `layers/positional_emb/rope.py`.
