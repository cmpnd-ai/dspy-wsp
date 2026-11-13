"""Strands decider (StrandsAgents/strands-decider-*, Apache-2.0): request encoding.

Every question gets its own prompt. The state comes first, so the questions of
one request share a prefix:

    <state>\\n{state}\\n</state>\\n
    <question type="{type}">\\n{header}\\n{instructions}\\n<options>\\n
    1. {name} — {description}\\n2. ...\\n</options>\\n</question>\\n<answer>

The head scores option k from the backbone's output at the last token of option
k's line, against its output at the last token of the prompt. This follows
`prompting.py` in the `strands-decider` package. The state and the question are
tokenized apart and joined, and a prompt over the window is refused, as there.
A noul's options are written as false, then true, so every type's options follow
`Question.option_keys`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.schema import Question, RequestError

NOUL_DEFAULTS = {"false": "the statement does not hold for this state", "true": "the statement holds for this state"}
HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}
MAX_LENGTH = 4096


@dataclass(frozen=True)
class Row:
    """One question's prompt: its tokens, how many of them are the state, and each option's last token."""

    ids: tuple[int, ...]
    state: int
    options: tuple[int, ...]
    kind: str


class Tokenizer:
    """The checkpoint's `tokenizer.json`, read with `tokenizers`."""

    def __init__(self, folder: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        self._tok = _Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
        self._tok.no_padding()
        self._tok.no_truncation()

    def encode(self, text: str, special: bool = False):
        return self._tok.encode(text, add_special_tokens=special)


def render(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


def options(q: Question) -> list[tuple[str, str]]:
    """(name, description) per option, in `Question.option_keys` order."""
    if q.type == "noul":
        criteria = {**NOUL_DEFAULTS, **(q.criteria or {})}
        return [(key, render(criteria[key])) for key in ("false", "true")]
    if q.type == "choice":
        return [(str(name), render(desc)) for name, desc in q.criteria.items()]
    return [(str(i), render(level)) for i, level in enumerate(q.criteria)]


def render_question(q: Question) -> tuple[str, list[tuple[int, int]]]:
    """The question's text and each option line's character span in it."""
    text = f'<question type="{q.type}">\n{HEADERS[q.type]}\n{render(q.instructions)}\n<options>\n'
    lines, spans = [], []
    at = len(text)
    for i, (name, desc) in enumerate(options(q)):
        desc = " ".join(desc.split())  # one option is one line
        line = f"{i + 1}. {name}" + (f" — {desc}" if desc else "")
        lines.append(line)
        spans.append((at, at + len(line)))
        at += len(line) + 1
    return text + "\n".join(lines) + "\n</options>\n</question>\n<answer>", spans


def encode(tok: Tokenizer, state: Any, q: Question, max_length: int = MAX_LENGTH) -> Row:
    state_ids = tok.encode(f"<state>\n{render(state)}\n</state>\n", special=True).ids
    text, spans = render_question(q)
    question = tok.encode(text)
    if len(state_ids) + len(question.ids) > max_length:
        raise RequestError(f"the prompt has {len(state_ids) + len(question.ids)} tokens ({len(state_ids)} state + "
                           f"{len(question.ids)} question); at most {max_length}")
    last = []
    for a, b in spans:
        inside = [j for j, (lo, hi) in enumerate(question.offsets) if hi > lo and lo >= a and hi <= b]
        if not inside:
            raise RequestError("an option has no tokens of its own")
        last.append(len(state_ids) + inside[-1])
    return Row(ids=tuple(state_ids + question.ids), state=len(state_ids), options=tuple(last), kind=q.type)


def probabilities(logits: np.ndarray, temperature: float) -> list[float]:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()
