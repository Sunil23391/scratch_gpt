import torch
import torch.nn as nn
from layers.normalization.rms_norm import RMSNorm
from layers.mlp.swiglu import SwiGLU
from layers.attention.attention import MultiHeadAttention
from layers.positional_emb.rope import RoPE

class ScratchTransformerModel(nn.Module):
    """Main Scratch Transformer assembly combining all modular layers."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        
        # Pre-normalization before the transformer blocks
        self.initial_norm = RMSNorm(cfg.dim, eps=cfg.eps)
        
        # Shared RoPE buffer
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
        # idx: [batch, seq_len]
        batch, seq_len = idx.shape
        
        # Word embeddings
        # x: [batch, seq_len, dim]
        x = self.embed(idx)
        x = self.initial_norm(x)
        
        for layer in self.layers:
            # Attention block (Pre-norm)
            # x_attn_norm: [batch, seq_len, dim]
            x_attn_norm = layer["attention_norm"](x)
            attn_out = layer["attention"](x_attn_norm, rope=self.rope) # <-- RoPE applied here
            x = x + attn_out
            
            # MLP block (Pre-norm)
            # x_mlp_norm: [batch, seq_len, dim]
            x_mlp_norm = layer["mlp_norm"](x)
            mlp_out = layer["mlp"](x_mlp_norm)
            x = x + mlp_out
            
        # Final output normalization and projection
        # x: [batch, seq_len, dim]
        x = self.final_norm(x)
        logits = self.lm_head(x) # [batch, seq_len, vocab_size]
        
        return logits
