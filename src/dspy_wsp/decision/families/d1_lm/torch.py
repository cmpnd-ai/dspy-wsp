"""The d1-2B network in PyTorch: the LFM3 language model, read at each answer position.

At a question's answer position the final normed hidden state is scored against
the token-embedding rows of the question's option tokens (the output layer is
tied to the embeddings). Each option keeps the best of its forms. Only those
rows are computed, never the full vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dspy_wsp.decision.families.d1_lm import encoding
from dspy_wsp.decision.families.d1_lm.encoding import Row
from dspy_wsp.decision.families.d1_lm.lfm3_torch import LanguageModel

TOKEN_BUDGET = 16_384  # padded tokens per forward
PREFIX = "model.language_model."


class D1LMTorch(nn.Module):
    def __init__(self, config: dict, pad_token_id: int = 0):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = LanguageModel(config["text_config"])
        self.window = config["text_config"]["sliding_window"]
        self.pad_token_id = pad_token_id

    @classmethod
    def from_dir(cls, folder: str | Path, device: str = "cpu", pad_token_id: int = 0) -> D1LMTorch:
        import json

        from safetensors.torch import load_file

        folder = Path(folder)
        net = cls(json.loads((folder / "config.json").read_text()), pad_token_id)
        weights = {k: v for k, v in load_file(str(folder / "model.safetensors")).items() if k.startswith(PREFIX)}
        net.load_state_dict(weights, strict=True)
        return net.to(device=device, dtype=torch.float32).eval()

    @property
    def device(self):
        return self.model.language_model.embed_tokens.weight.device

    @torch.no_grad()
    def _batch(self, rows: list[Row], layers: list[int]):
        width = max(len(r.ids) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r.ids)] = torch.tensor(r.ids)
        full, sliding, real = (torch.from_numpy(m).to(self.device) for m in encoding.masks(rows, width, self.window))
        h, kept = self.model.language_model(ids.to(self.device), full, sliding, real, layers)
        table = self.model.language_model.embed_tokens.weight
        out = []
        for i, r in enumerate(rows):
            recs = []
            for slot, groups in zip(r.slots, r.groups):
                x = h[i, slot]
                forms = [table[g] @ x for g in groups]
                best = [int(torch.argmax(f)) for f in forms]
                rec = {"logits": torch.stack([f[j] for f, j in zip(forms, best)]).cpu().numpy(),
                       "head_inputs": x[None].expand(len(groups), -1).cpu().numpy(),
                       "option_rows": torch.stack([table[g[j]] for g, j in zip(groups, best)]).cpu().numpy()}
                if layers:
                    rec["hidden_mean"] = torch.stack([kept[l][i, : r.state_len].mean(0) for l in layers]).cpu().numpy()
                recs.append(rec)
            out.append(recs)
        return out

    def _chunks(self, rows: list[Row]):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids), reverse=True)
        while order:
            size, longest = 1, len(rows[order[0]].ids)
            while size < len(order) and (size + 1) * longest <= TOKEN_BUDGET:
                size += 1
            yield order[:size]
            order = order[size:]

    def capture(self, rows: list[Row], layers: list[int]) -> list[list[dict[str, np.ndarray]]]:
        """Per row, per question: option logits, the answer position's hidden state (`head_inputs`, one copy per
        option), each option's best-form embedding row, and the hidden state after each listed layer averaged
        over the state's tokens."""
        out: list = [None] * len(rows)
        for idx in self._chunks(rows):
            for i, recs in zip(idx, self._batch([rows[i] for i in idx], list(layers))):
                out[i] = recs
        return out

    def logits(self, rows: list[Row]) -> list[list[np.ndarray]]:
        return [[rec["logits"] for rec in recs] for recs in self.capture(rows, [])]
