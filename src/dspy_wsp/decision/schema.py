"""System One requests and answers, in TypeSafe's shapes.

A request is `{"state": ..., "questions": {name: question}}`. A question has a
`type` (choice, noul or score), `instructions` and, for choice and score,
`criteria`. Answers carry the fields TypeSafe returns, with TypeSafe's
published confidence formulas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dspy_wsp.forward import RequestError

__all__ = ["Question", "RequestError", "choice_confidence", "make_answer", "parse_question", "parse_request", "score_confidence"]

MAX_CHOICES = 255
MAX_LEVELS = 10
_NOUL_KEYS = {"true": "true", "false": "false", "yes": "true", "no": "false"}


@dataclass(frozen=True)
class Question:
    type: str
    instructions: str
    criteria: Any = None  # choice: {key: description}; score: [level, ...]; noul: {"true"?, "false"?} or None

    @property
    def option_keys(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["false", "true"]


def parse_question(q: Any) -> Question:
    if not isinstance(q, Mapping):
        raise RequestError("a question must be an object with `type`, `instructions` and, for choice and score, `criteria`")
    kind = q.get("type")
    instructions = q.get("instructions", "")
    if not isinstance(instructions, str):
        raise RequestError("`instructions` must be a string")
    criteria = q.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= MAX_CHOICES:
            raise RequestError(f"a choice needs `criteria` with 2 to {MAX_CHOICES} options")
        if not all(isinstance(k, str) and k for k in criteria):
            raise RequestError("choice option keys must be non-empty strings")
        return Question("choice", instructions, dict(criteria))
    if kind == "score":
        if not isinstance(criteria, Sequence) or isinstance(criteria, str) or not 2 <= len(criteria) <= MAX_LEVELS:
            raise RequestError(f"a score needs `criteria`: a list of 2 to {MAX_LEVELS} levels, lowest first")
        return Question("score", instructions, list(criteria))
    if kind == "noul":
        if criteria is None:
            return Question("noul", instructions, None)
        if not isinstance(criteria, Mapping) or not set(criteria) <= set(_NOUL_KEYS):
            raise RequestError('noul `criteria` may only have the keys "true" and "false" (or "yes" and "no")')
        return Question("noul", instructions, {_NOUL_KEYS[k]: v for k, v in criteria.items()})
    raise RequestError(f"question `type` must be choice, noul or score, not {kind!r}")


def parse_request(body: Any) -> tuple[Any, dict[str, Question]]:
    if not isinstance(body, Mapping):
        raise RequestError("the request must be a JSON object")
    questions = body.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("`questions` must be a non-empty object")
    parsed = {}
    for name, q in questions.items():
        if not isinstance(name, str) or not name:
            raise RequestError("question names must be non-empty strings")
        parsed[name] = parse_question(q)
    return body.get("state", ""), parsed


def choice_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    return max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    m = max(range(n), key=probs.__getitem__)
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / mad_uniform)


def make_answer(q: Question, probs: Sequence[float]) -> dict:
    """The answer to `q`, from probabilities ordered as `q.option_keys`."""
    keys = q.option_keys
    probs = [float(p) for p in probs]
    if len(probs) != len(keys) or not all(math.isfinite(p) and p >= 0 for p in probs) or abs(sum(probs) - 1) > 1e-4:
        raise ValueError(f"expected a distribution over {len(keys)} options, got {probs}")
    if q.type == "noul":
        return {"type": "noul", "noul": probs[1]}
    distribution = dict(zip(keys, probs))
    if q.type == "choice":
        return {"type": "choice", "choice": max(keys, key=distribution.get), "probabilities": distribution, "confidence": choice_confidence(probs)}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "probabilities": distribution,
            "confidence": score_confidence(probs), "legend": {str(i): level for i, level in enumerate(q.criteria)}}
