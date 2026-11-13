"""Clef (Cloudflare/clef-flash, Apache-2.0): request encoding.

The state and every question of a request go into one ChatML prompt, and the
backbone reads it once:

    <|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n{state}
    \\n\\nSCHEMA FIELDS:\\n
    \\nFIELD 1\\nID: {name}\\nTYPE: {type}\\nINSTRUCTION: {instructions}\\nALLOWED OPTIONS:\\n
    OPTION 1: {"description":...,"option_id":...}\\n ... END FIELD\\n
    \\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA DECISIONS:

The head reads the backbone's output over each question's instruction tokens and
over each option's JSON tokens. This follows `encode_record` in the checkpoint's
`joint_schema_model.py`. A choice's options are written in sorted key order and
a noul's as true, then false; the spans here are returned in `Question.option_keys`
order, so logits need no reordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.schema import Question, RequestError

SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_DEFAULTS = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
MAX_LENGTH = 16_384

Span = tuple[int, int]


@dataclass(frozen=True)
class Field:
    """One question's token spans in the prompt. `options` follows `Question.option_keys`."""

    type_id: int
    question: Span
    options: tuple[Span, ...]


@dataclass(frozen=True)
class Row:
    """One request: its prompt's tokens, the span of the state's tokens and a `Field` per question."""

    ids: tuple[int, ...]
    state: Span
    fields: tuple[Field, ...]


class Tokenizer:
    """The checkpoint's `tokenizer.json`, read with `tokenizers` (no remote code)."""

    def __init__(self, folder: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        self._tok = _Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def written_options(q: Question) -> list[tuple[str, Any]]:
    """(option id, description) in the order the prompt lists them."""
    if q.type == "noul":
        criteria = {**NOUL_DEFAULTS, **(q.criteria or {})}
        return [(key, criteria[key]) for key in ("true", "false")]
    if q.type == "choice":
        return sorted((str(key), value) for key, value in q.criteria.items())
    return [(str(i), value) for i, value in enumerate(q.criteria)]


def encode_request(tok: Tokenizer, state: Any, questions: dict[str, Question], max_length: int = MAX_LENGTH) -> Row:
    schema = tok.encode("\n\nSCHEMA FIELDS:\n")
    fields = []
    for index, (name, q) in enumerate(questions.items()):
        schema += tok.encode(f"\nFIELD {index + 1}\nID: {name}\nTYPE: {q.type}\nINSTRUCTION: ")
        start = len(schema)
        schema += tok.encode(render(q.instructions or str(name)))
        question = (start, len(schema))
        schema += tok.encode("\nALLOWED OPTIONS:\n")
        spans = {}
        for i, (option_id, description) in enumerate(written_options(q)):
            schema += tok.encode(f"OPTION {i + 1}: ")
            start = len(schema)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema += tok.encode(render(semantics))
            spans[option_id] = (start, len(schema))
            schema += tok.encode("\n")
        schema += tok.encode("END FIELD\n")
        fields.append((QUESTION_TYPES[q.type], question, [spans[key] for key in q.option_keys]))
    prefix, suffix = tok.encode(PREFIX), tok.encode(SUFFIX)
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise RequestError(f"the questions need {fixed} tokens before the state; at most {max_length}")
    state_ids = tok.encode(render(state))[: max_length - fixed]  # the reference truncates the state to fit
    offset = len(prefix) + len(state_ids)
    shift = lambda span: (span[0] + offset, span[1] + offset)  # noqa: E731
    return Row(
        ids=tuple(prefix + state_ids + schema + suffix),
        state=(len(prefix), offset),
        fields=tuple(Field(type_id, shift(question), tuple(shift(s) for s in options)) for type_id, question, options in fields),
    )


def probabilities(logits: np.ndarray) -> list[float]:
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()
