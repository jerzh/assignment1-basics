from __future__ import annotations

import torch
import torch.nn as nn

from cs336_basics.modules import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLU,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # Spec state-dict keys (per adapter): ln1.weight, attn.*, ln2.weight, ffn.*
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope=True, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: torch.Tensor,                          # (..., seq_len, d_model)
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.ln1.forward(x), token_positions)
        return x + self.ffn.forward(self.ln2.forward(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # Spec state-dict keys:
        #   token_embeddings.weight       (vocab_size, d_model)
        #   layers.{i}.* (block keys)
        #   ln_final.weight               (d_model,)
        #   lm_head.weight                (vocab_size, d_model)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        in_indices: torch.Tensor,    # (batch, seq_len) int
    ) -> torch.Tensor:               # (batch, seq_len, vocab_size) logits
        batch, seq_len = in_indices.shape
        x = self.token_embeddings.forward(in_indices)
        positions = torch.arange(seq_len, device=x.device).expand(batch, seq_len)
        for block in self.layers:
            x = block(x, positions)
        x = self.ln_final.forward(x)
        return self.lm_head.forward(x)
