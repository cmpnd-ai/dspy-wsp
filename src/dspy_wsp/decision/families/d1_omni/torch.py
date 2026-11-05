"""The d1 text network in PyTorch, with module names equal to the checkpoint's tensor names.

The trunk is LFM2.5-Encoder-350M: LFM2 blocks (short convolutions and GQA
attention) made bidirectional, with a centred 3-tap convolution. The head adds a
question-type embedding, runs pre-norm transformer layers over the text and
scores each option marker with one MLP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dspy_wsp.decision.families.d1_omni import encoding as d1
from dspy_wsp.decision.schema import Question

TOKEN_BUDGET = 32_768  # padded tokens per forward


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


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

    def forward(self, x, cos, sin, mask):
        b, length, _ = x.shape
        shape = (b, length, -1, self.head_dim)
        q = self.q_layernorm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        return self.out_proj(y.transpose(1, 2).reshape(b, length, -1))


class ShortConv(nn.Module):
    """`out(C * conv(B * x))` with a centred 3-tap depthwise convolution."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.conv = nn.Conv1d(d, d, cfg["conv_L_cache"], groups=d, bias=False, padding=cfg["conv_L_cache"] - 1)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, pad):
        b, c, u = self.in_proj(x * pad[:, :, None]).transpose(-1, -2).chunk(3, dim=-2)
        bx = b * u
        w = self.conv.weight[:, 0, :]
        length = bx.shape[-1]
        xp = F.pad(bx, (1, 1))
        y = xp[..., 0:length] * w[:, 0, None] + xp[..., 1 : 1 + length] * w[:, 1, None] + xp[..., 2 : 2 + length] * w[:, 2, None]
        return self.out_proj((c * y).transpose(-1, -2))


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, hidden = cfg["hidden_size"], d1.mlp_width(cfg)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.rope_theta = cfg["rope_theta"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    def rope(self, length: int, x):
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim
        inv_freq = (1.0 / (self.rope_theta**exponent)).to(x.device)
        positions = torch.arange(length, device=x.device).float()
        freqs = (inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]

    def forward(self, h, pad, keep: list[int] = ()):
        """`h` (B, L, D) embeddings and `pad` (B, L), True on real tokens. Returns the normed output and the
        hidden states at the indices in `keep` (0 = embeddings, l = after block l-1)."""
        neg = max(-1e9, torch.finfo(h.dtype).min)
        mask = ((~pad).to(h.dtype) * neg)[:, None, None, :]
        padf = pad.to(h.dtype)
        cos, sin = self.rope(h.shape[1], h)
        kept = {0: h} if 0 in keep else {}
        for i, layer in enumerate(self.layers):
            x = layer.operator_norm(h)
            x = layer.self_attn(x, cos, sin, mask) if layer.is_attention_layer else layer.conv(x, padf)
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
            if i + 1 in keep:
                kept[i + 1] = h
        return self.embedding_norm(h), kept


class DecisionHead(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, d // 64, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, pad, marker_pos, qtype):
        """Returns the scorer's inputs at the markers (B, K, D) and the option logits (B, K)."""
        h = h + self.type_emb(qtype)[:, None, :]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=~pad)
        g = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        return g, self.scorer(g).squeeze(-1).float()


Row = tuple[list[int], list[int], Question]  # token ids, marker positions, question


class D1Torch(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        text = config["text_config"]
        self.encoder = Trunk(text)
        self.head = DecisionHead(text["hidden_size"], config["head_layers"])
        self.pad_token_id = int(config.get("pad_token_id", 0))

    @classmethod
    def from_dir(cls, folder: str | Path, device: str = "cpu") -> D1Torch:
        from safetensors.torch import load_file

        folder = Path(folder)
        model = cls(d1.load_config(folder))
        weights = {k: v for k, v in load_file(str(folder / "model.safetensors")).items() if d1.text_tensor(k)}
        model.load_state_dict(weights, strict=True)
        return model.to(device=device, dtype=torch.float32).eval()

    @property
    def device(self):
        return self.head.type_emb.weight.device

    @torch.no_grad()
    def _batch(self, rows: list[Row], keep: list[int]):
        device = self.device
        lengths = [len(ids) for ids, _, _ in rows]
        width, k = max(lengths), max(len(m) for _, m, _ in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        marker_pos = torch.zeros(len(rows), k, dtype=torch.long)
        for i, (row_ids, markers, _) in enumerate(rows):
            ids[i, : len(row_ids)] = torch.tensor(row_ids)
            marker_pos[i, : len(markers)] = torch.tensor(markers)
        ids, marker_pos = ids.to(device), marker_pos.to(device)
        pad = torch.arange(width, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        qtype = torch.tensor([d1.QTYPES[q.type] for _, _, q in rows], device=device)
        h, kept = self.encoder(self.encoder.embed_tokens(ids), pad, keep)
        g, logits = self.head(h, pad, marker_pos, qtype)
        trunk_markers = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        padf = pad.float()[:, :, None]
        means = {l: (kept[l] * padf).sum(1) / padf.sum(1) for l in keep}
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
            for j, i in enumerate(idx):
                out[i] = z[j, : len(rows[i][1])].cpu().numpy()
        return out

    def capture(self, rows: list[Row], layers: list[int]) -> list[dict[str, np.ndarray]]:
        """Per row: option logits, the scorer's inputs and the trunk output at each marker, and the
        hidden state after each layer in `layers` averaged over the row's tokens."""
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            z, g, t, means = self._batch([rows[i] for i in idx], list(layers))
            for j, i in enumerate(idx):
                k = len(rows[i][1])
                rec = {"logits": z[j, :k].cpu().numpy(), "head_inputs": g[j, :k].cpu().numpy(), "trunk_markers": t[j, :k].cpu().numpy()}
                if layers:
                    rec["hidden_mean"] = torch.stack([means[l][j] for l in layers]).cpu().numpy()
                out[i] = rec
        return out
