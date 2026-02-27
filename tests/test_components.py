import torch
from layers.normalization.rms_norm import RMSNorm
from layers.mlp.swiglu import SwiGLU
from layers.positional_emb.rope import RoPE
from layers.attention.attention import MultiHeadAttention
from model import ScratchTransformerModel
from config import Config

def test_rms_norm():
    print("Testing RMSNorm...")
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    # RMSNorm should keep mean squared equal to 1 (approx) before scaling
    # We just check shape and lack of NaNs here
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

    # Second token
    out2 = attn(x, use_cache=True)
    assert out2.shape == (2, 1, dim)
    assert len(attn.kv_cache.keys) == 2
    
    attn.kv_cache.reset()
    assert len(attn.kv_cache.keys) == 0
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
