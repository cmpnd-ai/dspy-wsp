"""The NanoJev network in MLX (see `nanojev_torch` for the architecture). The backbone is
mlx-lm's Qwen3 model; parameter names equal the checkpoint's tensor names.

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision import nanojev
from dspy_wsp.decision.d1_mlx import MultiheadAttention
from dspy_wsp.decision.nanojev_torch import TOKEN_BUDGET, Row


class NanoJevMLX(nn.Module):
    def __init__(self, backbone_config: dict, set_head: str = "attention", pad_token_id: int = 0):
        super().__init__()
        from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

        self.backbone = Qwen3Model(ModelArgs.from_dict(backbone_config))
        hidden = backbone_config["hidden_size"]
        self.norm = nn.LayerNorm(hidden, eps=1e-5)
        self.scalar = nn.Linear(hidden, 1)
        self._set_head = set_head
        if set_head == "attention":
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = MultiheadAttention(128, 4)
            self.set_output = nn.Linear(128, 1)
        self._pad = pad_token_id

    @classmethod
    def from_dir(cls, folder: str | Path, pad_token_id: int = 0) -> NanoJevMLX:
        run, backbone = nanojev.load_configs(folder)
        model = cls(backbone, run.get("set_head", "attention"), pad_token_id)
        weights = {k: v.astype(mx.float32) for k, v in mx.load(str(Path(folder) / "best.safetensors")).items()}
        model.load_named(weights, strict=True)
        return model

    def load_named(self, weights: dict, strict: bool = False) -> None:
        self.load_weights(list(weights.items()), strict=strict)
        mx.eval(self.parameters())

    def _backbone(self, tokens, keep: list[int]):
        from mlx_lm.models.base import create_attention_mask

        h = self.backbone.embed_tokens(tokens)
        mask = create_attention_mask(h, None)
        kept = {0: h} if 0 in keep else {}
        for i, layer in enumerate(self.backbone.layers):
            h = layer(h, mask, None)
            if i + 1 in keep:
                kept[i + 1] = h
        return self.backbone.norm(h), kept

    def _batch(self, rows: list[Row], keep: list[int]):
        paths = [p for r in rows for p in r[0]]
        lengths = np.array([len(p) for p in paths])
        tokens = np.full((len(paths), int(lengths.max())), self._pad, dtype=np.int32)
        for i, p in enumerate(paths):
            tokens[i, : len(p)] = p
        last, kept = self._backbone(mx.array(tokens), keep)
        ends = last[mx.arange(len(paths)), mx.array(lengths - 1)]
        mx.eval(ends, *kept.values())
        records, offset = [], 0
        for paths_r, prefix, q in rows:
            n = len(paths_r)
            leaves = ends[offset : offset + n]
            h = self.norm(leaves)
            z = self.scalar(h)[:, 0]
            delta = mx.zeros((n,))
            if q.type == "choice" and self._set_head == "attention":
                log_k = mx.full((1, n, 1), math.log(n))
                u = self.set_project(mx.concatenate([h[None], log_k], axis=-1))
                delta = self.set_output(mx.tanh(u + self.set_attention(u, None)))[0, :, 0]
                z = z + delta
            logits = mx.stack([z[0] * 0, z[0]]) if q.type == "noul" else z
            rec = {"logits": np.array(logits), "head_inputs": np.array(h), "path_ends": np.array(leaves), "set_delta": np.array(delta)}
            if keep:
                rec["hidden_mean"] = np.stack([np.array(kept[l][offset, :prefix].mean(axis=0)) if l < len(self.backbone.layers) else np.array(last[offset, :prefix].mean(axis=0)) for l in keep])
            records.append(rec)
            offset += n
        return records

    def _chunks(self, rows: list[Row]):
        order = sorted(range(len(rows)), key=lambda i: max(map(len, rows[i][0])), reverse=True)
        while order:
            size, longest = 1, max(map(len, rows[order[0]][0]))
            while size < len(order) and sum(len(rows[i][0]) for i in order[: size + 1]) * longest <= TOKEN_BUDGET:
                size += 1
            yield order[:size]
            order = order[size:]

    def capture(self, rows: list[Row], layers: list[int]) -> list[dict[str, np.ndarray]]:
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            for i, rec in zip(idx, self._batch([rows[i] for i in idx], list(layers))):
                out[i] = rec
        return out

    def logits(self, rows: list[Row]) -> list[np.ndarray]:
        return [rec["logits"] for rec in self.capture(rows, [])]
