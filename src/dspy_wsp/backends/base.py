"""What WeightSpace needs from a model backend.

A backend owns the student model's weights. It must:
  * expose the original weights as safetensors files (`weights_dir`) for the kernel and patches;
  * swap edited tensors in and out (`set_weights`);
  * provide a `dspy.LM` whose calls run on the current weights (`lm`), recording each call;
  * run teacher-forced reads that return log-probs and internals (`teacher_forced`, `pooled_activations`).
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LMCall:
    prompt: str
    completion: str
    finish_reason: str
    input_tokens: int
    output_tokens: int


class WeightBackend(ABC):
    def __init__(self) -> None:
        self._edits: dict[str, np.ndarray] = {}
        self._recording: list[LMCall] | None = None

    # --- identity -------------------------------------------------------------------
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def weights_dir(self) -> Path:
        """Local folder with the original safetensors, config and tokenizer files."""

    def describe(self) -> str:
        """A short architecture description for the writer's task prompt."""
        return self.model_id

    # --- weights --------------------------------------------------------------------
    @abstractmethod
    def _apply(self, edits: Mapping[str, np.ndarray]) -> None:
        """Restore the original weights, then apply `edits` (float32 arrays keyed by checkpoint name)."""

    def set_weights(self, edits: Mapping[str, np.ndarray] | None) -> None:
        edits = dict(edits or {})
        self._apply(edits)
        self._edits = edits

    @property
    def current_edits(self) -> dict[str, np.ndarray]:
        return dict(self._edits)

    @contextlib.contextmanager
    def using(self, edits: Mapping[str, np.ndarray] | None):
        previous = self.current_edits
        self.set_weights(edits)
        try:
            yield self
        finally:
            self.set_weights(previous)

    # --- generation -----------------------------------------------------------------
    @abstractmethod
    def lm(self):
        """A dspy.LM whose calls run on the backend's current weights. Must not cache."""

    @abstractmethod
    def seed(self, seed: int) -> None:
        """Fix the sampling seed for the calls that follow (derived per call)."""

    @contextlib.contextmanager
    def record_calls(self):
        calls: list[LMCall] = []
        self._recording = calls
        try:
            yield calls
        finally:
            self._recording = None

    def _record(self, call: LMCall) -> None:
        if self._recording is not None:
            self._recording.append(call)

    @abstractmethod
    def render_chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = False, **template_kwargs) -> str: ...

    # --- reading internals ----------------------------------------------------------
    @abstractmethod
    def teacher_forced(
        self,
        texts: Sequence[str],
        *,
        token_logprobs: bool = False,
        topk: int = 0,
        hidden_states: Sequence[int] = (),
        mlp_act: Sequence[int] = (),
        attn_layers: Sequence[int] = (),
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Read texts without generating. Returns (arrays, meta)."""

    @abstractmethod
    def pooled_activations(
        self, pairs: Sequence[tuple[str, str]], *, layers: Sequence[int], mlp_layers: Sequence[int] = (), max_answer_tokens: int = 8192
    ) -> dict[str, np.ndarray]:
        """For each (prompt, completion): states at the prompt's last token and averaged over the completion."""

    def editable_names(self) -> list[str]:
        """Checkpoint tensors the backend can swap in."""
        from dspy_wsp.tensors import TensorReader

        return TensorReader(self.weights_dir).names

    def dims(self) -> dict:
        """Sizes used to estimate how much a teacher-forced pass will write: layers, hidden, intermediate, heads."""
        return {}

    @abstractmethod
    def num_tokens(self, text: str) -> int: ...

    def close(self) -> None:
        pass
