"""The LFM3 language model in MLX. Parameter names equal the checkpoint's, and the
computation matches `lfm3_torch.py` in this folder, which describes the architecture.

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision.families.d1_lm.lfm3_torch import QUERY_CHUNK, inverse_frequencies

NEG = -1e9


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return self.weight * (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps))


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, f = cfg["hidden_size"], cfg["intermediate_size"]
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


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

    def __call__(self, x, cos, sin, bias):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, -1, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, t, -1, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, -1, self.head_dim).transpose(0, 2, 1, 3)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        if groups > 1:
            k, v = mx.repeat(k, groups, axis=1), mx.repeat(v, groups, axis=1)
        y = mx.concatenate([mx.fast.scaled_dot_product_attention(q[:, :, s : s + QUERY_CHUNK], k, v, scale=self.head_dim**-0.5,
                                                                 mask=bias[:, :, s : s + QUERY_CHUNK])
                            for s in range(0, t, QUERY_CHUNK)], axis=2)
        gate = mx.sigmoid(self.gate_proj(x)).transpose(0, 2, 1)[..., None]
        return self.o_proj((y * gate).transpose(0, 2, 1, 3).reshape(b, t, -1))


class _ConvWeight(nn.Module):
    def __init__(self, d: int, taps: int):
        super().__init__()
        self.weight = mx.zeros((d, 1, taps))  # PyTorch's Conv1d layout


class ShortConv(nn.Module):
    """y = out_proj(C * conv(B * x)): a gated, causal, depthwise convolution."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.taps = cfg["conv_L_cache"]
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = _ConvWeight(d, self.taps)
        self.out_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, real):
        keep = real[:, :, None]
        b, c, h = mx.split(self.in_proj(x * keep), 3, axis=-1)
        bx = b * h * keep
        t = bx.shape[1]
        w = self.conv.weight[:, 0, :]
        xp = mx.pad(bx, [(0, 0), (self.taps - 1, 0), (0, 0)])
        y = sum(xp[:, j : j + t] * w[:, j] for j in range(self.taps))
        return self.out_proj(c * y)


class Layer(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        self.kind = kind
        d, eps = cfg["hidden_size"], cfg["norm_eps"]
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

    def __call__(self, h, rope, full, sliding, real):
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
        self._cfg = cfg
        d = cfg["hidden_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], d)
        self.layers = [Layer(cfg, kind) for kind in cfg["layer_types"]]
        self.embedding_norm = RMSNorm(d, cfg["norm_eps"])
        head_dim = cfg.get("head_dim") or d // cfg["num_attention_heads"]
        self._inv = {kind: inverse_frequencies(head_dim, p["rope_theta"], p.get("partial_rotary_factor", 1.0)).numpy()
                     for kind, p in cfg.get("rope_parameters", {}).items() if kind in cfg["layer_types"]}

    def __call__(self, ids, full, sliding, real, layers=()):
        """The final normed hidden states, and the hidden state after each listed layer (0 = embeddings).
        `full` and `sliding` are boolean [B, T, T] arrays; `real` is [B, T]."""
        h = self.embed_tokens(ids)
        kept = {0: h} if 0 in layers else {}
        t = ids.shape[1]
        pos = np.arange(t, dtype=np.float32)
        rope = {}
        for kind, inv in self._inv.items():
            freqs = pos[:, None] * inv[None]
            emb = np.concatenate([freqs, freqs], axis=-1)
            rope[kind] = (mx.array(np.cos(emb))[None, None], mx.array(np.sin(emb))[None, None])
        to_bias = lambda m: mx.array(np.where(m, 0.0, NEG).astype(np.float32))[:, None]  # noqa: E731
        full_bias, sliding_bias = to_bias(full), to_bias(sliding)
        real = mx.array(real.astype(np.float32))
        for i, layer in enumerate(self.layers):
            h = layer(h, rope, full_bias, sliding_bias, real)
            if i + 1 in layers:
                kept[i + 1] = h
        return self.embedding_norm(h), kept
