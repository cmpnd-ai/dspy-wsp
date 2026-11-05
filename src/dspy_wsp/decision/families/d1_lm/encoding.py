"""d1-2B (LiquidAI/d1-2B): how a request becomes token ids, and how answers are read.

d1-2B is an LFM3 language model that answers in one forward pass. Each request is
one ChatML user turn: the state, then up to 16 questions packed after it. Each
question ends with the assistant header and an empty reasoning block; the
model's next-token scores at that position, restricted to the tokens that code
the question's options, are the answer:

    <|startoftext|><|im_start|>user\\n{state}\\n\\n\\nQUESTION:\\n
    {question 1}<|im_end|>\\n<|im_start|>assistant\\n<think></think>      <- answer position 1
    QUESTION:\\n{question 2}<|im_end|>\\n<|im_start|>assistant\\n<think></think>   <- answer position 2

A choice's options are coded A, B, C..., a score's levels 0, 1, 2..., a noul is
yes or no. This follows Liquid's `prompt.py`, `attention.py` and `runner.py` at
the pinned revision, for text states and these three question types.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.schema import Question, RequestError

SET_SIZE = 16  # questions read together after one state, the trained size
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
LEAD = "<think></think>"  # what the chat template writes before a reply without reasoning
HEADER = "QUESTION:\n"
BREAK = "​"  # written after a special token's first character, so state text cannot form the token
SUPPORTED = ("choice", "noul", "score")
YES_FORMS, NO_FORMS = ("yes", "Yes", "YES"), ("no", "No", "NO")
LETTERS = tuple(chr(ord("A") + i) for i in range(26))
# Codes tried in order when an option's own code is not a single unused token.
FALLBACK_POOL = (
    list(LETTERS)
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [a + b for a in LETTERS for b in LETTERS]
)


class Tokenizer:
    """The checkpoint's `tokenizer.json`, read with `tokenizers` (no remote code)."""

    def __init__(self, folder: str | Path):
        from tokenizers import Tokenizer as _Tokenizer

        folder = Path(folder)
        self._tok = _Tokenizer.from_file(str(folder / "tokenizer.json"))
        cfg = json.loads((folder / "tokenizer_config.json").read_text())

        def content(v):
            return v["content"] if isinstance(v, dict) else v

        self.bos = content(cfg.get("bos_token")) or ""
        self.pad_token_id = self._tok.token_to_id(content(cfg.get("pad_token")) or "<|pad|>") or 0
        added = self._tok.get_added_tokens_decoder().values()
        self.specials = sorted({t.content for t in added if t.special and len(t.content) > 1 and BREAK not in t.content},
                               key=len, reverse=True)
        self.pattern = re.compile("|".join(map(re.escape, self.specials))) if self.specials else None

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def single(self, texts: Sequence[str]) -> list[int]:
        """The ids of the texts that are one token each, without repeats."""
        out: list[int] = []
        for t in texts:
            ids = self.encode(t)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out


def plain(tok: Tokenizer, text: str) -> str:
    """`text` with every special-token string broken, so it encodes as ordinary text."""
    if tok.pattern is None:
        return text
    n = 1
    while n:  # again, for a match that overlapped one broken before it
        text, n = tok.pattern.subn(lambda m: m[0][0] + BREAK + m[0][1:], text)
    return text


def state_block(state: Any) -> str:
    """A string state as written, any other state as indented JSON; then a blank line."""
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def prefix(tok: Tokenizer, state: Any) -> str:
    return f"{tok.bos}{IM_START}user\n{plain(tok, state_block(state))}\n{HEADER}"


# --- options -----------------------------------------------------------------------------

def option_codes(labels: Sequence[str], scheme: int = 1) -> list[str]:
    """The labels themselves when all are single letters; else A..Z; past 26, 00.. (scheme 1) or A..Z, AA, AB.. (2)."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return list(LETTERS[: len(labs)])
    if scheme == 2:
        return (list(LETTERS) + [a + b for a in LETTERS for b in LETTERS])[: len(labs)]
    return [f"{i:02d}" for i in range(len(labs))]


def code_tokens(tok: Tokenizer, keys: Sequence[str], scheme: int = 2) -> list[tuple[str, int]]:
    """Each option's code and its token: its own code if that is one unused token, else the next free fallback."""
    used: set[int] = set()
    out: list[tuple[str, int]] = []

    def take(raw: str) -> bool:
        ids = tok.encode(raw)
        if len(ids) != 1 or ids[0] in used:
            return False
        out.append((raw, ids[0]))
        used.add(ids[0])
        return True

    for code in option_codes(keys, scheme):
        if not take(code) and not any(take(raw) for raw in FALLBACK_POOL):
            raise RequestError(f"no single-token code left for {len(keys)} options")
    return out


def readout_groups(tok: Tokenizer, q: Question, scheme: int = 2) -> list[list[int]]:
    """The tokens scored for each option, in `q.option_keys` order; an option scores its best form."""
    if q.type == "noul":
        yes, no = tok.single(YES_FORMS), tok.single(NO_FORMS)
        if not yes or not no:
            raise RequestError("the tokenizer has no single-token yes or no")
        return [no, yes]  # option keys are [false, true]
    if q.type == "score":
        groups = [tok.single([str(i)]) for i in range(len(q.criteria))]
        if not all(groups):
            raise RequestError("a score needs a single-token digit per level")
        return groups
    groups = []
    for code, tid in code_tokens(tok, list(q.criteria), scheme):
        groups.append([tid] + [i for i in tok.single([f" {code}"]) if i != tid])
    return groups


# --- questions ---------------------------------------------------------------------------

def check_supported(raw: Any) -> None:
    """Refuse question types d1-2B answers in other shapes (text, number, date, box, multiple choice)."""
    if isinstance(raw, Question):
        return
    if isinstance(raw, Mapping) and (raw.get("type") not in SUPPORTED or raw.get("multiple")):
        raise RequestError("d1-2B in dspy-wsp answers choice, noul and score questions (single choice only)")


def question_block(tok: Tokenizer, q: Question, scheme: int = 2) -> str:
    if q.type == "choice":
        keys = list(q.criteria)
        lines = "\n".join(f"{code} {q.criteria[key] or key.replace('_', ' ')}"
                          for (code, _), key in zip(code_tokens(tok, keys, scheme), keys))
        return f"{q.instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if q.type == "noul":
        extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}" if q.criteria else ""
        return f"{q.instructions}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {level}" for i, level in enumerate(q.criteria))
    return f"{q.instructions}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."


def suffix(tok: Tokenizer, q: Question, scheme: int = 2) -> str:
    """The question and the assistant header, up to its answer position."""
    return f"{plain(tok, question_block(tok, q, scheme))}{IM_END}\n{IM_START}assistant\n{LEAD}"


# --- rows --------------------------------------------------------------------------------

@dataclass
class Row:
    """One sequence: a state and the questions packed after it."""

    ids: list[int]
    state_len: int  # the state's tokens, which read the whole sequence both ways
    question_of: list[int]  # per token: -1 on the state, k on question k (1-based)
    slots: list[int]  # each question's answer position
    questions: list[Question]
    groups: list[list[list[int]]] = field(default_factory=list)  # per question: the tokens each option scores


def encode_request(tok: Tokenizer, state: Any, questions: Sequence[Question], *, max_length: int,
                   set_size: int = SET_SIZE, scheme: int = 2) -> list[Row]:
    """The request's rows: its questions in order, `set_size` to a row after a copy of the state."""
    head = prefix(tok, state)
    suffixes = [suffix(tok, q, scheme) for q in questions]
    rows = []
    for start in range(0, len(questions), set_size):
        first = suffixes[start]
        ids = tok.encode(head + first)
        state_len = len(ids) - len(tok.encode(first))
        question_of = [-1] * state_len + [1] * (len(ids) - state_len)
        slots = [len(ids) - 1]
        for k, text in enumerate(suffixes[start + 1 : start + set_size], 2):
            own = tok.encode(HEADER + text)
            ids += own
            question_of += [k] * len(own)
            slots.append(len(ids) - 1)
        if len(ids) > max_length:
            raise RequestError(f"a request row has {len(ids)} tokens; at most {max_length}")
        own = list(questions[start : start + set_size])
        rows.append(Row(ids, state_len, question_of, slots, own, [readout_groups(tok, q, scheme) for q in own]))
    return rows


# --- answers -----------------------------------------------------------------------------

def probabilities(scores: np.ndarray, temperature: float) -> list[float]:
    """A softmax of the option scores divided by the calibration temperature."""
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()


def bucket(q: Question) -> str:
    """The calibration bucket of a question: its type, and for a score its levels, for a choice its size."""
    if q.type == "noul":
        return "noul"
    k = len(q.criteria)
    if q.type == "score":
        return f"score:{k}"
    return "choice:small" if k <= 3 else "choice:mid" if k <= 8 else "choice:large" if k <= 26 else "choice:inventory"


@dataclass
class Calibration:
    """`calibration.json`: a temperature per bucket where one was fitted, else the default."""

    temperatures: dict[str, float] = field(default_factory=dict)
    default: float = 1.0

    def temperature(self, q: Question) -> float:
        return float(self.temperatures.get(bucket(q), self.default))


def load_calibration(folder: str | Path) -> Calibration:
    f = Path(folder) / "calibration.json"
    if not f.is_file():
        return Calibration()
    d = json.loads(f.read_text())
    return Calibration(dict(d.get("temperatures") or {}), float(d.get("default", 1.0)))


# --- batching ----------------------------------------------------------------------------

TOKEN_BUDGET = 16_384  # padded tokens per forward
MASK_BUDGET = 1 << 24  # query-key cells per mask per forward: rows x width x width


def batches(rows: Sequence[Row]):
    """Row indices in batches, longest first, within both budgets; a row over a budget runs alone."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids), reverse=True)
    while order:
        size, longest = 1, len(rows[order[0]].ids)
        while size < len(order) and (size + 1) * longest <= TOKEN_BUDGET and (size + 1) * longest**2 <= MASK_BUDGET:
            size += 1
        yield order[:size]
        order = order[size:]


# --- attention ---------------------------------------------------------------------------

def masks(rows: Sequence[Row], width: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which keys each query reads, [rows, query, key], for the two kinds of attention layer; and the real tokens.

    Full attention: a state token reads every token, every token reads the state, and a question's tokens
    read each other both ways. Sliding-window attention: causal, over the last `window` tokens.
    Convolution layers are causal and need only `real`.
    """
    b = len(rows)
    real = np.zeros((b, width), bool)
    group = np.full((b, width), -1)
    state = np.zeros((b, 1, 1), int)
    for r, row in enumerate(rows):
        real[r, : len(row.ids)] = True
        group[r, : len(row.ids)] = row.question_of
        state[r] = row.state_len
    q = np.arange(width)[None, :, None]
    k = np.arange(width)[None, None, :]
    keys = real[:, None, :]
    full = keys & ((q < state) | (k < state) | (group[:, :, None] == group[:, None, :]))
    sliding = keys & (k <= q) & (k > q - window)
    return full, sliding, real
