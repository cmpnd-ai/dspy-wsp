"""The d1-2B network in MLX. It matches `torch.py` in this folder, which describes the readout.

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision.families.d1_lm import encoding
from dspy_wsp.decision.families.d1_lm.encoding import Row
from dspy_wsp.decision.families.d1_lm.lfm3_mlx import LanguageModel
from dspy_wsp.decision.families.d1_lm.torch import PREFIX


class _Model(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.language_model = LanguageModel(cfg)


class D1LMMLX(nn.Module):
    def __init__(self, config: dict, pad_token_id: int = 0):
        super().__init__()
        self.model = _Model(config["text_config"])
        self._window = config["text_config"]["sliding_window"]
        self._pad = pad_token_id

    @classmethod
    def from_dir(cls, folder: str | Path, pad_token_id: int = 0) -> D1LMMLX:
        folder = Path(folder)
        net = cls(json.loads((folder / "config.json").read_text()), pad_token_id)
        weights = {k: v.astype(mx.float32) for k, v in mx.load(str(folder / "model.safetensors")).items() if k.startswith(PREFIX)}
        net.load_named(weights, strict=True)
        return net

    def load_named(self, weights: dict, strict: bool = False) -> None:
        self.load_weights(list(weights.items()), strict=strict)
        mx.eval(self.parameters())

    def _batch(self, rows: list[Row], layers: list[int]):
        width = max(len(r.ids) for r in rows)
        ids = np.full((len(rows), width), self._pad, dtype=np.int32)
        for i, r in enumerate(rows):
            ids[i, : len(r.ids)] = r.ids
        full, sliding, real = encoding.masks(rows, width, self._window)
        lm = self.model.language_model
        h, kept = lm(mx.array(ids), full, sliding, real, layers)
        table = lm.embed_tokens.weight
        mx.eval(h, *kept.values())
        out = []
        for i, r in enumerate(rows):
            recs = []
            for slot, groups in zip(r.slots, r.groups):
                x = h[i, slot]
                forms = [np.array(table[mx.array(g)] @ x) for g in groups]
                best = [int(np.argmax(f)) for f in forms]
                rows_used = mx.stack([table[g[j]] for g, j in zip(groups, best)])
                rec = {"logits": np.array([f[j] for f, j in zip(forms, best)], np.float32),
                       "head_inputs": np.repeat(np.array(x)[None], len(groups), axis=0),
                       "option_rows": np.array(rows_used)}
                if layers:
                    rec["hidden_mean"] = np.stack([np.array(kept[l][i, : r.state_len].mean(axis=0)) for l in layers])
                recs.append(rec)
            out.append(recs)
        return out

    def _chunks(self, rows: list[Row]):
        return encoding.batches(rows)

    def capture(self, rows: list[Row], layers: list[int]) -> list[list[dict[str, np.ndarray]]]:
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            for i, recs in zip(idx, self._batch([rows[i] for i in idx], list(layers))):
                out[i] = recs
        return out

    def logits(self, rows: list[Row]) -> list[list[np.ndarray]]:
        return [[rec["logits"] for rec in recs] for recs in self.capture(rows, [])]
