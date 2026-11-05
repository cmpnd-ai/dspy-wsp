"""NanoJev (C-Tianyu/NanoJev, source TianyuCodings/NanoJev, MIT): request encoding.

Every option is read on its own text path, and the backbone's hidden state at
the path's last token (EOS) is that option's input to the decision head:

    State:\\n{state}\\n  Question type: {type}\\nQuestion:\\n{instructions}\\n  Candidate:\\n{text}\\nDecision:<eos>

A noul (NanoJev's `boolean`) has one path, with text "The proposition is true.";
its logits are [0, z]. A score's candidates are its level descriptions. This
follows `prepare_examples` in NanoJev's `scripts/predict_toy_decisions.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.schema import Question, RequestError

BOOLEAN_TEXT = "The proposition is true."
NATIVE_TYPE = {"choice": "choice", "score": "score", "noul": "boolean"}


class Tokenizer:
    """The checkpoint's `tokenizer/tokenizer.json`, read with `tokenizers` (no remote code)."""

    def __init__(self, folder: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        folder = Path(folder)
        self._tok = _Tokenizer.from_file(str(folder / "tokenizer.json"))
        cfg = json.loads((folder / "tokenizer_config.json").read_text())
        eos = cfg.get("eos_token")
        eos = eos["content"] if isinstance(eos, dict) else eos
        self.eos_token_id = self._tok.token_to_id(eos)
        pad = cfg.get("pad_token") or eos
        pad = pad["content"] if isinstance(pad, dict) else pad
        self.pad_token_id = self._tok.token_to_id(pad)
        if self.eos_token_id is None or self.pad_token_id is None:
            raise ValueError("the NanoJev tokenizer needs eos and pad tokens")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


def candidate_texts(q: Question) -> list[str]:
    if q.type == "choice":
        # NanoJev's reference requires a description for every option; an option without one is read as its key.
        return [k if v is None or v == "" else f"{k}: {v}" for k, v in q.criteria.items()]
    if q.type == "score":
        return [f"{c}" for c in q.criteria]
    return [BOOLEAN_TEXT]


def encode(tok: Tokenizer, state: Any, q: Question, max_len: int) -> tuple[list[list[int]], int]:
    """One token path per candidate, and the length of the prefix they share (state and question)."""
    segments = [f"State:\n{state}\n", f"Question type: {NATIVE_TYPE[q.type]}\nQuestion:\n{q.instructions}\n"]
    if q.type == "noul" and q.criteria:
        for key, label in (("false", "False"), ("true", "True")):
            if key in q.criteria:
                segments[1] += f"{label} criterion: {q.criteria[key]}\n"
    prefix = sum((tok.encode(s) for s in segments), [])
    paths = [prefix + tok.encode(f"Candidate:\n{t}\nDecision:") + [tok.eos_token_id] for t in candidate_texts(q)]
    longest = max(map(len, paths))
    if longest > max_len:
        raise RequestError(f"a candidate path has {longest} tokens; at most {max_len} (NanoJev does not truncate)")
    return paths, len(prefix)


def boolean_logits(z: float) -> np.ndarray:
    """A noul's logits over [false, true] from the head's single output."""
    return np.array([0.0, z], dtype=np.float32)


def probabilities(logits: np.ndarray, q: Question) -> list[float]:
    z = np.asarray(logits, dtype=np.float64)[: len(q.option_keys)]
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()


def load_configs(folder: str | Path) -> tuple[dict, dict]:
    """(run config, backbone config) of a NanoJev checkpoint folder."""
    folder = Path(folder)
    return json.loads((folder / "config.json").read_text()), json.loads((folder / "backbone_config" / "config.json").read_text())
