"""The NanoJev network in PyTorch, with module names equal to the checkpoint's tensor names.

The backbone is a Qwen3 model. Each candidate path's last hidden state goes
through a LayerNorm (`norm`) and a scalar head (`scalar`). For choice questions
a set-attention correction over the options (`set_project`, `set_attention`,
`set_output`) adds to each option's logit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dspy_wsp.decision.families.nanojev import encoding as nanojev
from dspy_wsp.decision.schema import Question

TOKEN_BUDGET = 32_768

Row = tuple[list[list[int]], int, Question]  # candidate paths, shared prefix length, question


class NanoJevTorch(nn.Module):
    def __init__(self, backbone_config: dict, set_head: str = "attention", pad_token_id: int = 0):
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model

        cfg = Qwen3Config.from_dict({k: v for k, v in backbone_config.items() if k not in ("architectures", "auto_map")})
        cfg.use_cache = False
        cfg._attn_implementation = "sdpa"
        self.backbone = Qwen3Model(cfg)
        hidden = cfg.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)
        self.set_head = set_head
        if set_head == "attention":
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
        self.pad_token_id = pad_token_id

    @classmethod
    def from_dir(cls, folder: str | Path, device: str = "cpu", pad_token_id: int = 0) -> NanoJevTorch:
        from safetensors.torch import load_file

        run, backbone = nanojev.load_configs(folder)
        model = cls(backbone, run.get("set_head", "attention"), pad_token_id)
        model.load_state_dict(load_file(str(Path(folder) / "best.safetensors")), strict=True)
        return model.to(device=device, dtype=torch.float32).eval()

    @property
    def device(self):
        return self.scalar.weight.device

    @torch.no_grad()
    def _batch(self, rows: list[Row], keep: list[int]):
        device = self.device
        paths = [p for r in rows for p in r[0]]
        lengths = torch.tensor([len(p) for p in paths], device=device)
        tokens = torch.full((len(paths), int(lengths.max())), self.pad_token_id, dtype=torch.long, device=device)
        for i, p in enumerate(paths):
            tokens[i, : len(p)] = torch.tensor(p, device=device)
        attention = torch.arange(tokens.shape[1], device=device)[None] < lengths[:, None]
        out = self.backbone(input_ids=tokens, attention_mask=attention, use_cache=False, output_hidden_states=bool(keep))
        ends = out.last_hidden_state[torch.arange(len(paths), device=device), lengths - 1]
        records, offset = [], 0
        for paths_r, prefix, q in rows:
            n = len(paths_r)
            leaves = ends[offset : offset + n]
            h = self.norm(leaves)
            z = self.scalar(h).squeeze(-1).float()
            delta = torch.zeros(n, device=device)
            if q.type == "choice" and self.set_head == "attention":
                log_k = torch.full((1, n, 1), math.log(n), device=device, dtype=h.dtype)
                u = self.set_project(torch.cat([h[None], log_k], dim=-1))
                mixed, _ = self.set_attention(u, u, u, need_weights=False)
                delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1)[0].float()
                z = z + delta
            logits = torch.stack([z[0] * 0, z[0]]) if q.type == "noul" else z
            rec = {"logits": logits.cpu().numpy(), "head_inputs": h.cpu().numpy(), "path_ends": leaves.cpu().numpy(), "set_delta": delta.cpu().numpy()}
            if keep:
                rec["hidden_mean"] = torch.stack([out.hidden_states[l][offset, :prefix].mean(0) for l in keep]).cpu().numpy()
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
