"""The d1 text network in MLX. Parameter names equal the checkpoint's tensor names,
and the computation matches `d1_torch` (see that module for the architecture).

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision import d1
from dspy_wsp.decision.d1_torch import TOKEN_BUDGET, Row

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


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = d // self.heads
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])

    def __call__(self, x, cos, sin, mask):
        b, length, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).reshape(b, length, -1, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_layernorm(self.k_proj(x).reshape(b, length, -1, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, length, -1, self.head_dim).transpose(0, 2, 1, 3)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        if groups > 1:
            k, v = mx.repeat(k, groups, axis=1), mx.repeat(v, groups, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(b, length, -1))


class _ConvWeight(nn.Module):
    def __init__(self, d: int, taps: int):
        super().__init__()
        self.weight = mx.zeros((d, 1, taps))


class ShortConv(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.conv = _ConvWeight(d, cfg["conv_L_cache"])
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, pad):
        b, c, u = mx.split(self.in_proj(x * pad[:, :, None]), 3, axis=-1)  # (B, L, D) each
        bx = b * u
        w = self.conv.weight[:, 0, :]  # (D, 3)
        xp = mx.pad(bx, [(0, 0), (1, 1), (0, 0)])
        length = bx.shape[1]
        y = xp[:, 0:length] * w[:, 0] + xp[:, 1 : 1 + length] * w[:, 1] + xp[:, 2 : 2 + length] * w[:, 2]
        return self.out_proj(c * y)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, hidden = cfg["hidden_size"], d1.mlp_width(cfg)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, cfg: dict, kind: str):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(cfg)
        else:
            self.conv = ShortConv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])


class Trunk(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [Layer(cfg, kind) for kind in cfg["layer_types"]]
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self._rope_theta = cfg["rope_theta"]
        self._head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    def rope(self, length: int):
        exponent = np.arange(0, self._head_dim, 2, dtype=np.int64).astype(np.float32) / self._head_dim
        inv_freq = (1.0 / (self._rope_theta**exponent)).astype(np.float32)
        freqs = np.arange(length, dtype=np.float32)[:, None] * inv_freq[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return mx.array(np.cos(emb))[None, None], mx.array(np.sin(emb))[None, None]

    def __call__(self, h, pad, keep=()):
        mask = ((1 - pad) * NEG)[:, None, None, :]
        cos, sin = self.rope(h.shape[1])
        kept = {0: h} if 0 in keep else {}
        for i, layer in enumerate(self.layers):
            x = layer.operator_norm(h)
            x = layer.self_attn(x, cos, sin, mask) if layer.is_attention_layer else layer.conv(x, pad)
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
            if i + 1 in keep:
                kept[i + 1] = h
        return self.embedding_norm(h), kept


class MultiheadAttention(nn.Module):
    """`torch.nn.MultiheadAttention` (batch first, packed input projection) for self-attention."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * d, d))
        self.in_proj_bias = mx.zeros((3 * d,))
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x, mask):
        b, length, d = x.shape
        q, k, v = mx.split(x @ self.in_proj_weight.T + self.in_proj_bias, 3, axis=-1)
        hd = d // self.heads
        q, k, v = (t.reshape(b, length, self.heads, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=mask)
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(b, length, d))


class EncoderLayer(nn.Module):
    """`torch.nn.TransformerEncoderLayer` with norm_first=True, ReLU and no dropout."""

    def __init__(self, d: int):
        super().__init__()
        self.self_attn = MultiheadAttention(d, d // 64 or 1)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class _Stack(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.layers = [EncoderLayer(d) for _ in range(layers)]


class DecisionHead(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        self.head = _Stack(d, layers)
        self.scorer = [nn.LayerNorm(d, eps=1e-5), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)]

    def __call__(self, h, pad, marker_pos, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        mask = ((1 - pad) * NEG)[:, None, None, :]
        for layer in self.head.layers:
            h = layer(h, mask)
        g = mx.take_along_axis(h, marker_pos[:, :, None], axis=1)
        z = g
        for f in self.scorer:
            z = f(z)
        return g, z[..., 0]


class D1MLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        text = config["text_config"]
        self.encoder = Trunk(text)
        self.head = DecisionHead(text["hidden_size"], config["head_layers"])
        self._pad = int(config.get("pad_token_id", 0))

    @property
    def config(self) -> dict:
        return self._config

    @classmethod
    def from_dir(cls, folder: str | Path) -> D1MLX:
        folder = Path(folder)
        model = cls(d1.load_config(folder))
        weights = {k: v.astype(mx.float32) for k, v in mx.load(str(folder / "model.safetensors")).items() if d1.text_tensor(k)}
        model.load_named(weights, strict=True)
        return model

    def load_named(self, weights: dict, strict: bool = False) -> None:
        self.load_weights(list(weights.items()), strict=strict)
        mx.eval(self.parameters())

    def _batch(self, rows: list[Row], keep: list[int]):
        lengths = [len(ids) for ids, _, _ in rows]
        width, k = max(lengths), max(len(m) for _, m, _ in rows)
        ids = np.full((len(rows), width), self._pad, dtype=np.int32)
        marker_pos = np.zeros((len(rows), k), dtype=np.int32)
        for i, (row_ids, markers, _) in enumerate(rows):
            ids[i, : len(row_ids)] = row_ids
            marker_pos[i, : len(markers)] = markers
        pad = mx.array((np.arange(width)[None] < np.array(lengths)[:, None]).astype(np.float32))
        qtype = mx.array([d1.QTYPES[q.type] for _, _, q in rows])
        h, kept = self.encoder(self.encoder.embed_tokens(mx.array(ids)), pad, keep)
        mp = mx.array(marker_pos)
        g, logits = self.head(h, pad, mp, qtype)
        trunk_markers = mx.take_along_axis(h, mp[:, :, None], axis=1)
        p3 = pad[:, :, None]
        means = {l: (kept[l] * p3).sum(axis=1) / p3.sum(axis=1) for l in keep}
        mx.eval(logits, g, trunk_markers, *means.values())
        return logits, g, trunk_markers, means

    def _chunks(self, rows: list[Row]):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]), reverse=True)
        while order:
            size, longest = 1, len(rows[order[0]][0])
            while size < len(order) and (size + 1) * longest <= TOKEN_BUDGET:
                size += 1
            yield order[:size]
            order = order[size:]

    def logits(self, rows: list[Row]) -> list[np.ndarray]:
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            z, *_ = self._batch([rows[i] for i in idx], [])
            z = np.array(z)
            for j, i in enumerate(idx):
                out[i] = z[j, : len(rows[i][1])]
        return out

    def capture(self, rows: list[Row], layers: list[int]) -> list[dict[str, np.ndarray]]:
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            z, g, t, means = self._batch([rows[i] for i in idx], list(layers))
            z, g, t = np.array(z), np.array(g), np.array(t)
            means = {l: np.array(v) for l, v in means.items()}
            for j, i in enumerate(idx):
                k = len(rows[i][1])
                rec = {"logits": z[j, :k], "head_inputs": g[j, :k], "trunk_markers": t[j, :k]}
                if layers:
                    rec["hidden_mean"] = np.stack([means[l][j] for l in layers])
                out[i] = rec
        return out
