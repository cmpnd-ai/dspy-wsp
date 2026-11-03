"""What WeightSpace needs from a model backend.

A backend owns the student model's weights. It must:
  * expose the original weights as safetensors files (`weights_dir`) for the kernel and patches;
  * swap edited tensors in and out (`set_weights`);
  * provide a `dspy.LM` whose calls run on the current weights (`lm`), recording each call;
  * run teacher-forced reads that return log-probs and internals (`teacher_forced`, `pooled_activations`).
"""

from __future__ import annotations

import contextlib
import threading
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
    # How many program runs the forward-pass tool may have in flight at once.
    # Batched backends raise this so concurrent LM calls share a batch.
    concurrency: int = 1

    def __init__(self) -> None:
        self._edits: dict[str, np.ndarray] = {}
        self._local = threading.local()  # per-thread seed and call recording

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

    def seed(self, seed: int) -> None:
        """Fix the sampling seed for this thread's calls that follow (derived per call)."""
        self._local.seed = int(seed)
        self._local.call_index = 0

    def _next_seed(self, explicit: int | None = None) -> int | None:
        if explicit is not None:
            return int(explicit)
        base = getattr(self._local, "seed", None)
        if base is None:
            return None
        s = (base * 1_000_003 + self._local.call_index) % (2**63 - 1)
        self._local.call_index += 1
        return s

    @contextlib.contextmanager
    def record_calls(self):
        """Collect the LM calls this thread makes inside the block."""
        calls: list[LMCall] = []
        self._local.calls = calls
        try:
            yield calls
        finally:
            self._local.calls = None

    def _record(self, call: LMCall) -> None:
        calls = getattr(self._local, "calls", None)
        if calls is not None:
            calls.append(call)

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


class BackendEngine:
    """A DSPy 3.4 engine: one canonical request in, one response out, run on the backend's current weights.

    Subclasses set `backend_class`. DSPy saves a program's LM through `dump_state`
    (the base model, its settings and the patch) and rebuilds it with `load_state`.
    """

    backend_class: type | None = None

    def __init__(self, backend):
        self.backend = backend

    def complete(self, request):
        from dspy.clients.engines.base import validate_request
        from dspy.lm15 import Message, Response, TextPart, ThinkingPart, Usage

        validate_request(request)
        msgs = []
        if request.system is not None:
            system = request.system if isinstance(request.system, str) else "".join(getattr(p, "text", "") or "" for p in request.system)
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.text or ""} for m in request.messages)
        prompt = self.backend.render_chat(msgs, add_generation_prompt=True)
        c = request.config
        call = self.backend.generate(prompt, max_tokens=c.max_tokens, temperature=c.temperature, top_p=c.top_p, top_k=c.top_k, seed=c.seed, stop=c.stop)
        text = call.completion
        parts = [TextPart(text)]
        if self.backend.split_thinking and "</think>" in text:
            thinking, answer = text.split("</think>", 1)
            parts = [ThinkingPart(thinking.replace("<think>", "").strip()), TextPart(answer.lstrip())]
        elif self.backend.split_thinking and (prompt.rstrip().endswith("<think>") or text.lstrip().startswith("<think>")):
            # Cut off while thinking: there is no answer yet.
            parts = [ThinkingPart(text.replace("<think>", "").strip()), TextPart("")]
        return Response(
            id=None,
            model=self.backend.model_id,
            message=Message.assistant(parts),
            finish_reason=call.finish_reason,
            usage=Usage(input_tokens=call.input_tokens, output_tokens=call.output_tokens, total_tokens=call.input_tokens + call.output_tokens),
        )

    def stream(self, request):
        from dspy.lm15 import response_to_events

        return response_to_events(self.complete(request))

    def close(self) -> None:
        pass

    def dump_state(self) -> dict:
        return {"backend": self.backend._init_kwargs, "patch": self.backend.patch_dir}

    @classmethod
    def load_state(cls, state: dict):
        # DSPy loads each predictor's LM separately (and once more in a trial run),
        # so share one backend per (class, model, settings, patch).
        import json

        key = cls.__name__ + json.dumps(state, sort_keys=True, default=str)
        backend = LOADED_BACKENDS.get(key)
        if backend is None:
            from dspy_wsp.patch import Patch

            backend = cls.backend_class(**state["backend"])
            if state.get("patch"):
                patch = Patch.load(state["patch"])
                backend.set_weights({n: t.float().numpy() for n, t in patch.edited_tensors(backend.reader).items()})
                backend.patch_dir = state["patch"]
            LOADED_BACKENDS[key] = backend
        return cls(backend)


LOADED_BACKENDS: dict[str, WeightBackend] = {}
