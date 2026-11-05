"""Liquid AI's d1 decision models: request encoding, shared by the PyTorch and MLX networks.

Each question is read against the state as one sequence:

    <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> ... <decide>

The head scores the hidden state at every `<mask>`, and a softmax over the
question's options, after a calibrated temperature, gives the probabilities.
This follows the text path of `prompt.py` in `LiquidAI/d1-omni-600M` (LFM Open
License v1.0); the audio and image paths are not supported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.schema import Question

QTYPES = {"choice": 0, "score": 1, "noul": 2}
DELIM = {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>", "opt_end": "<|reserved_10|>", "decide": "<|reserved_11|>"}
MARKER = "<|mask|>"
PER_OPTION = 24
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Tokenizer:
    """The checkpoint's `tokenizer.json`, read with `tokenizers` (no remote code)."""

    def __init__(self, folder: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        folder = Path(folder)
        self._tok = _Tokenizer.from_file(str(folder / "tokenizer.json"))
        cfg = json.loads((folder / "tokenizer_config.json").read_text())
        self.bos_token_id = self.token_id(_token_text(cfg.get("bos_token")))
        self.pad_token_id = self.token_id(_token_text(cfg.get("pad_token")))

    def token_id(self, token: str) -> int:
        i = self._tok.token_to_id(token)
        if i is None:
            raise ValueError(f"the tokenizer has no token {token!r}")
        return i

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def _token_text(t: Any) -> str:
    return t["content"] if isinstance(t, dict) else t


def escape(text: str) -> str:
    """`<|name|>` becomes `<¦name¦>`, so caller text cannot produce a delimiter or marker token."""
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _criterion(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Question) -> list[str]:
    """Option texts in the model's order. A noul is read as [false, true]."""
    if q.type == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in q.criteria.items()]
    if q.type == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(q.criteria)]
    crit = q.criteria or {}
    false, true = crit.get("false"), crit.get("true")
    return [
        "false: " + (_criterion(false) if false not in (None, "") else "no, the statement does not hold"),
        "true: " + (_criterion(true) if true not in (None, "") else "yes, the statement holds"),
    ]


def encode(tok: Tokenizer, state: Any, q: Question, max_len: int) -> tuple[list[int], list[int]]:
    """Token ids of one question over one state, and the position of each option's marker.

    The option block gets max(96, min(24k + 32, max_len / 2)) tokens, shared out
    evenly; the state is truncated on the right to the room that is left.
    """
    enc = lambda s: tok.encode(escape(s))  # noqa: E731
    opts = render_options(q)
    budget = max(96, min(len(opts) * PER_OPTION + 32, max_len // 2))
    per = max(2, (budget - 3 * len(opts)) // len(opts))
    question = ([tok.token_id(DELIM["q"])] + enc(q.instructions))[: max(16, budget)]
    markers = []
    for text in opts:
        markers.append(len(question) + 1)
        question += [tok.token_id(DELIM["opt"]), tok.token_id(MARKER)] + enc(" " + text)[:per] + [tok.token_id(DELIM["opt_end"])]
    question.append(tok.token_id(DELIM["decide"]))
    room = max(0, max_len - len(question) - 2)
    state_ids = [tok.token_id(DELIM["state"])] + enc(serialize(state))[:room]
    ids = ([tok.bos_token_id] + state_ids + question)[:max_len]
    markers = [m + 1 + len(state_ids) for m in markers]
    if markers[-1] >= max_len:
        raise ValueError("the options do not fit in the context")
    return ids, markers


def temperature_key(q: Question) -> str:
    k = len(q.option_keys)
    return f"{q.type}:" + ("2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+")


def temperature(q: Question, temperatures: dict) -> float:
    return float(temperatures.get(temperature_key(q), temperatures.get(q.type, 1.0)))


def probabilities(logits: np.ndarray, q: Question, temperatures: dict) -> list[float]:
    """The calibrated distribution over `q.option_keys` (the model's own option order)."""
    z = np.asarray(logits, dtype=np.float64)[: len(q.option_keys)] / temperature(q, temperatures)
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()


def load_config(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "config.json").read_text())


def mlp_width(cfg: dict) -> int:
    hidden = int(2 * cfg["intermediate_size"] / 3)
    hidden = int(cfg["block_ffn_dim_multiplier"] * hidden)
    m = cfg["block_multiple_of"]
    return m * ((hidden + m - 1) // m)


def text_tensor(name: str) -> bool:
    """Tensors of the text path; the audio and vision encoders are not loaded."""
    return name.startswith(("encoder.", "head."))
