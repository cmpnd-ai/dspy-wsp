"""The LFM3 language model in PyTorch, with module names equal to the checkpoint's.

A stack of blocks, each a gated short convolution, a sliding-window attention or
a full attention, then a SwiGLU MLP. Every block is pre-normed, and with
`use_sandwich_norm` its two outputs are normed again before they join the
residual stream. Attention normalizes queries and keys per head, rotates them
(RoPE with its own theta per attention kind; a partial rotation leaves the tail
dimensions unrotated) and gates each head's output with a sigmoid. The final
norm is `embedding_norm`.

Masks come from `encoding.masks`: full-attention layers read `full`, sliding
layers read `sliding`, and convolutions are causal over the real tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUERY_CHUNK = 512  # queries per attention call, to bound the score matrix


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def inverse_frequencies(head_dim: int, theta: float, partial: float) -> torch.Tensor:
    """RoPE frequencies of the full head dimension; past `partial` of it, zero (unrotated)."""
    angles = int(partial * head_dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / head_dim))
    return torch.cat((inv, torch.zeros(head_dim // 2 - angles)))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, f = cfg["hidden_size"], cfg["intermediate_size"]
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or d // self.heads
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.gate_proj = nn.Linear(d, self.heads, bias=False)

    def forward(self, x, cos, sin, bias):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, self.head_dim).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        out = torch.cat([F.scaled_dot_product_attention(q[:, :, s : s + QUERY_CHUNK], k, v, attn_mask=bias[:, None, s : s + QUERY_CHUNK],
                                                        scale=self.head_dim**-0.5)
                         for s in range(0, t, QUERY_CHUNK)], dim=2)
        gate = torch.sigmoid(self.gate_proj(x)).transpose(1, 2)[..., None]
        return self.o_proj((out * gate).transpose(1, 2).reshape(b, t, -1))


class ShortConv(nn.Module):
    """y = out_proj(C * conv(B * x)): a gated, causal, depthwise convolution."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, width, bias = cfg["hidden_size"], cfg["conv_L_cache"], cfg.get("conv_bias", False)
        self.in_proj = nn.Linear(d, 3 * d, bias=bias)
        self.conv = nn.Conv1d(d, d, kernel_size=width, groups=d, bias=bias, padding=width - 1)
        self.out_proj = nn.Linear(d, d, bias=bias)

    def forward(self, x, real):
        t = x.shape[1]
        keep = real[:, :, None].to(x.dtype)
        b, c, h = self.in_proj(x * keep).transpose(1, 2).chunk(3, dim=1)
        y = self.conv(b * h * keep.transpose(1, 2))[..., :t]
        return self.out_proj((c * y).transpose(1, 2))


class Layer(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        self.kind = kind
        eps = cfg["norm_eps"]
        d = cfg["hidden_size"]
        if kind == "conv":
            self.conv = ShortConv(cfg)
        else:
            self.self_attn = Attention(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(d, eps)
        self.ffn_norm = RMSNorm(d, eps)
        self.sandwich = cfg.get("use_sandwich_norm", False)
        if self.sandwich:
            self.post_operator_norm = RMSNorm(d, eps)
            self.post_ffn_norm = RMSNorm(d, eps)

    def forward(self, h, rope, full, sliding, real):
        x = self.operator_norm(h)
        if self.kind == "conv":
            y = self.conv(x, real)
        else:
            cos, sin = rope[self.kind]
            y = self.self_attn(x, cos, sin, full if self.kind == "full_attention" else sliding)
        if self.sandwich:
            y = self.post_operator_norm(y)
        h = h + y
        y = self.feed_forward(self.ffn_norm(h))
        if self.sandwich:
            y = self.post_ffn_norm(y)
        return h + y


class LanguageModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], d)
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.embedding_norm = RMSNorm(d, cfg["norm_eps"])
        self.head_dim = cfg.get("head_dim") or d // cfg["num_attention_heads"]

    def forward(self, ids, full, sliding, real, layers=()):
        """The final normed hidden states, and the hidden state after each listed layer (0 = embeddings)."""
        h = self.embed_tokens(ids).float()
        kept = {0: h} if 0 in layers else {}
        pos = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32)
        rope = {}
        for kind, p in self.cfg.get("rope_parameters", {}).items():
            if kind not in self.cfg["layer_types"]:
                continue
            inv = inverse_frequencies(self.head_dim, p["rope_theta"], p.get("partial_rotary_factor", 1.0)).to(ids.device)
            freqs = pos[:, None] * inv[None]
            emb = torch.cat((freqs, freqs), dim=-1)
            rope[kind] = (emb.cos()[None, None], emb.sin()[None, None])
        def to_bias(mask):  # 0 where a query reads a key, the most negative float elsewhere
            return torch.zeros(mask.shape, device=mask.device).masked_fill(~mask, torch.finfo(torch.float32).min)

        full, sliding = to_bias(full), to_bias(sliding)
        for i, layer in enumerate(self.layers):
            h = layer(h, rope, full, sliding, real)
            if i + 1 in layers:
                kept[i + 1] = h
        return self.embedding_norm(h), kept
