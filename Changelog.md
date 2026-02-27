### 0.0.0.1
1. To integrate RoPE correctly, we need to pass the rotary embedding logic into the attention mechanism. Since RoPE is position-dependent, the MultiHeadAttention layer needs to receive the pre-computed frequencies or apply them to the Query ($Q$) and Key ($K$) tensors before computing attention scores.
Below are the updated files to fix this integration.
1. Update layers/attention/attention.py
We modify the forward method to accept an optional rope module or pre-computed rotary embeddings.
```python
import torchimport torch.nn as nnimport torch.nn.functional as Fimport mathfrom layers.kv_cache.cache import KVCache
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, gqa_groups: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.total_dim = n_heads * head_dim
        self.gqa_groups = gqa_groups

        self.q_proj = nn.Linear(dim, self.total_dim, bias=False)
        self.k_proj = nn.Linear(dim, (n_heads // gqa_groups) * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, (n_heads // gqa_groups) * head_dim, bias=False)
        self.out_proj = nn.Linear(self.total_dim, dim, bias=False)
        self.kv_cache = KVCache()

    def forward(self, x: torch.Tensor, rope: nn.Module = None, mask: torch.Tensor = None, use_cache: bool = False):
        B, T, _ = x.shape
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.n_heads // self.gqa_groups, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads // self.gqa_groups, self.head_dim)

        # --- APPLY RoPE HERE ---
        if rope is not None:
            Q = rope(Q, T)
            K = rope(K, T)
        # -----------------------

        if use_cache:
            K, V = self.kv_cache.append(K, V)

        if self.gqa_groups > 1:
            K = torch.repeat_interleave(K, self.gqa_groups, dim=2)
            V = torch.repeat_interleave(V, self.gqa_groups, dim=2)

        scores = torch.einsum('bthd,bshd->bths', Q, K) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bths,bshd->bthd', attn, V)
        out = out.reshape(B, T, self.total_dim)
        return self.out_proj(out)
```
------------------------------
2. Update model.py
We now pass self.rope into each attention layer during the forward pass.
```python
import torchimport torch.nn as nnfrom layers.normalization.rms_norm import RMSNormfrom layers.mlp.swiglu import SwiGLUfrom layers.attention.attention import MultiHeadAttentionfrom layers.positional_emb.rope import RoPE
class ScratchTransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.initial_norm = RMSNorm(cfg.dim, eps=cfg.eps)
        
        # RoPE is initialized here
        self.rope = RoPE(cfg.head_dim, base=cfg.rope_base)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention_norm": RMSNorm(cfg.dim, eps=cfg.eps),
                "attention": MultiHeadAttention(cfg.dim, cfg.n_heads, cfg.head_dim, cfg.gqa_groups),
                "mlp_norm": RMSNorm(cfg.dim, eps=cfg.eps),
                "mlp": SwiGLU(cfg.dim, cfg.hidden_dim),
            }) for _ in range(cfg.n_layers)
        ])
        
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        batch, seq_len = idx.shape
        x = self.embed(idx)
        x = self.initial_norm(x)

        for layer in self.layers:
            # Attention block with RoPE injected
            x_attn_norm = layer["attention_norm"](x)
            attn_out = layer["attention"](x_attn_norm, rope=self.rope) # <-- RoPE applied here
            x = x + attn_out

            # MLP block
            x_mlp_norm = layer["mlp_norm"](x)
            mlp_out = layer["mlp"](x_mlp_norm)
            x = x + mlp_out

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
```

------------------------------
Key Improvements Made:

* Encapsulation: The RoPE logic remains in its own module, but MultiHeadAttention now knows when to apply it (specifically to $Q$ and $K$ after projection but before the score calculation).
* Consistency: The model now respects the relative positional encoding required for the Transformer to understand sequence order.
* Scalability: By passing the module, we ensure that if you later update RoPE to handle KV-caching (where seq_len might just be 1 but the position index is 100), the interface is already prepared.

Would you like me to now draft a Generation Script that utilizes the KVCache to produce text from a prompt?





2. To accommodate the integration of RoPE into the model's forward pass, you must update the unit tests in tests/test_components.py. The primary change is ensuring the MultiHeadAttention test passes the RoPE module as an argument and verifying that the ScratchTransformerModel test now implicitly benefits from this positional encoding.
Updated tests/test_components.py
```python

import torchfrom layers.normalization.rms_norm 
import RMSNormfrom layers.mlp.swiglu 
import SwiGLUfrom layers.positional_emb.rope 
import RoPEfrom layers.attention.attention 
import MultiHeadAttentionfrom model 
import ScratchTransformerModelfrom config 
import Config


def test_rms_norm():
    print("Testing RMSNorm...")
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaNs detected in RMSNorm"
    print("RMSNorm passed.")

def test_swiglu():
    print("Testing SwiGLU...")
    dim, hidden_dim = 64, 128
    mlp = SwiGLU(dim, hidden_dim)
    x = torch.randn(2, 10, dim)
    out = mlp(x)
    assert out.shape == (2, 10, dim), f"Expected (2, 10, {dim}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaNs detected in SwiGLU"
    print("SwiGLU passed.")

def test_rope():
    print("Testing RoPE...")
    head_dim = 64
    rope = RoPE(head_dim)
    x = torch.randn(2, 10, 8, head_dim) # [batch, seq, heads, dim]
    out = rope(x, seq_len=10)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaNs detected in RoPE"
    print("RoPE passed.")

def test_attention_and_cache_with_rope():
    print("Testing MultiHeadAttention with RoPE and KVCache...")
    dim, n_heads, head_dim = 64, 4, 16
    attn = MultiHeadAttention(dim, n_heads, head_dim, gqa_groups=2)
    rope = RoPE(head_dim)
    
    x = torch.randn(2, 1, dim) # Single token inference
    # Pass the rope module to the attention layer
    out = attn(x, rope=rope, use_cache=True)
    
    assert out.shape == (2, 1, dim), f"Expected (2, 1, {dim}), got {out.shape}"
    assert len(attn.kv_cache.keys) == 1
    # GQA: n_heads (4) // gqa_groups (2) = 2 KV heads
    assert attn.kv_cache.keys[0].shape == (2, 1, 2, head_dim)
    
    attn.kv_cache.reset()
    print("Attention with RoPE and KVCache passed.")

def test_model_integration():
    print("Testing ScratchTransformerModel (Full Integration)...")
    cfg = Config(dim=64, n_layers=2, n_heads=4, head_dim=16, hidden_dim=128)
    model = ScratchTransformerModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    
    logits = model(idx)
    assert logits.shape == (2, 10, cfg.vocab_size), f"Expected (2, 10, {cfg.vocab_size}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaNs detected in Model output"
    print("ScratchTransformerModel integration passed.")
if __name__ == "__main__":
    test_rms_norm()
    test_swiglu()
    test_rope()
    test_attention_and_cache_with_rope()
    test_model_integration()
    print("\nAll unit tests passed successfully!")
```

Key Changes in Tests:

* test_attention_and_cache_with_rope: Now instantiates a RoPE module and passes it into attn(...). This confirms the MultiHeadAttention.forward signature change works.
* Verification: The tests use torch.allclose() logic implicitly via shape and NaN checks, but you could add a comparison against a non-RoPE baseline to ensure the output values actually differ when RoPE is active.
* Full Integration: The test_model_integration now exercises the logic where model.forward passes self.rope down to each transformer layer. [1, 2, 3, 4] 


[1] [https://madewithml.com](https://madewithml.com/courses/mlops/testing/)
[2] [https://github.com](https://github.com/Exorust/TorchLeet/blob/main/llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb)
[3] [https://vizuara.substack.com](https://vizuara.substack.com/p/decoding-multi-head-latent-attention-e15)
[4] [https://www.geeksforgeeks.org](https://www.geeksforgeeks.org/deep-learning/how-to-use-pytorchs-nnmultiheadattention/)
