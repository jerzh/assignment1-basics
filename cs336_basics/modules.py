from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange, einsum, repeat


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(w, 0, sigma, -3*sigma, 3*sigma)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(w, 0, 1, -3, 3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        result = x / rms * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = self.w1.forward(x)
        return self.w2.forward(
            t1 * t1.sigmoid() * self.w3.forward(x)
        )


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp = (x - x.amax(dim, keepdim=True)).exp()
    return exp / exp.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,                          # (..., queries, d_k)
    K: torch.Tensor,                          # (..., keys,    d_k)
    V: torch.Tensor,                          # (..., keys,    d_v)
    mask: torch.Tensor | None = None,         # (..., queries, keys) bool; True = attend
) -> torch.Tensor:                            # (..., queries, d_v)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(K.shape[-1])
    scores = scores.masked_fill(~mask, -torch.inf)
    return einsum(softmax(scores, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        freqs = (1.0 / theta) ** (torch.arange(0, d_k, 2, device=device) / d_k)
        angles = torch.arange(max_seq_len, device=device).unsqueeze(1) * freqs
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,                  # (..., seq_len, d_k)
        token_positions: torch.Tensor,    # (..., seq_len)
    ) -> torch.Tensor:
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        out_even = cos * x_even - sin * x_odd
        out_odd  = sin * x_even + cos * x_odd
        return torch.stack((out_even, out_odd), dim=-1).flatten(start_dim=-2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

    def forward(
        self,
        x: torch.Tensor,                          # (..., seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # (..., seq_len)
    ) -> torch.Tensor:
        q = rearrange(self.q_proj.forward(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj.forward(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj.forward(x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        if self.use_rope:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)
        seq_len = q.shape[-2]
        assert k.shape[-2] == seq_len
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool, device=x.device))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.output_proj.forward(attn)
