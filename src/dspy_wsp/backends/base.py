"""What WeightSpace needs from a model backend.

A backend owns the student model's weights. Every backend (`WeightBackend`) must:
  * expose the original weights as safetensors files (`weights_dir`) for the kernel and patches;
  * swap edited tensors in and out (`_apply`, called by `set_weights`);
  * provide a `dspy.LM` whose calls run on the current weights (`lm`), recording each call;
  * count a text's tokens (`num_tokens`).

There are two kinds, and WeightSpace gives the writer different passes for each:
  * `GenerativeBackend` (kind "llm") generates text. It adds `generate`, `render_chat`
    and the teacher-forced reads (`teacher_forced`, `pooled_activations`).
  * `DecisionBackend` (kind "decision", in `dspy_wsp.backends.decision`) answers typed
    questions with probabilities. It adds `system_one` and `read`.

To write a backend, subclass one of the two and implement its abstract methods.
The other attributes and methods here have defaults.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from dspy_wsp.edits import DEFAULT_PROTECTED


@dataclass
class LMCall:
    prompt: str
    completion: str
    finish_reason: str
    input_tokens: int
    output_tokens: int


class WeightBackend(ABC):
    # "llm" or "decision": which passes and task prompt the writer gets.
    kind: ClassVar[str] = "llm"
    # How many program runs the forward-pass tool may have in flight at once.
    # Batched backends raise this so concurrent LM calls share a batch.
    concurrency: int = 1
    # The checkpoint's revision on the Hugging Face Hub, recorded in the patch.
    revision: str | None = None
    # The folder of the patch whose edits the backend holds. WeightSpace.compile sets it, and a saved program records it.
    patch_dir: str | None = None
    # How the task prompt names the protected tensors. None uses the standard sentence.
    protected_description: str | None = None

    def __init__(self) -> None:
        self._edits: dict[str, np.ndarray] = {}
        self._local = threading.local()  # per-thread seed and call recording
        self._init_kwargs: dict = {}
        self._reader = None

    def init_kwargs(self) -> dict:
        """The constructor arguments that rebuild this backend. A saved program stores them."""
        return self._init_kwargs

    @property
    def reader(self):
        """A TensorReader over the original weights in `weights_dir`."""
        if self._reader is None:
            from dspy_wsp.tensors import TensorReader

            self._reader = TensorReader(self.weights_dir)
        return self._reader

    @reader.setter
    def reader(self, value) -> None:
        self._reader = value

    @property
    def tokenizer_dir(self) -> Path | None:
        """A folder with `tokenizer.json` for the writer's `tok`. None loads the tokenizer from `weights_dir` with transformers."""
        return None

    def default_protected(self) -> tuple[str, ...]:
        """Regexes of tensor names the writer may read but not edit, used when WeightSpace gets no `protected`."""
        return DEFAULT_PROTECTED

    def describe_sampling(self) -> str:
        """The sampling settings, for the writer's task prompt."""
        return "the backend's defaults"

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

    def editable_names(self) -> list[str]:
        """Checkpoint tensors the backend can swap in."""
        return self.reader.names

    def dims(self) -> dict:
        """The network's sizes: layers, hidden, intermediate, heads. They bound layer indices and size a pass's results."""
        return {}

    @abstractmethod
    def num_tokens(self, text: str) -> int: ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GenerativeBackend(WeightBackend):
    """A backend for a text-generating LLM."""

    kind: ClassVar[str] = "llm"
    # Return text before "</think>" as reasoning, not as the answer.
    split_thinking: bool = True
    # Sampling settings, described to the writer. None leaves a setting out.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    chat_template_kwargs: dict | None = None

    def describe_sampling(self) -> str:
        parts = [f"{name} {v}" for name in ("temperature", "top_p", "top_k") if (v := getattr(self, name)) is not None]
        if self.max_tokens is not None:
            parts.append(f"at most {self.max_tokens} new tokens per LM call")
        if self.chat_template_kwargs:
            parts.append(f"chat template options {self.chat_template_kwargs}")
        return ", ".join(parts) or super().describe_sampling()

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None, stop=()) -> LMCall:
        """Complete one rendered prompt on the current weights, and record the call."""

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


class BackendEngine:
    """A DSPy engine: one canonical request in, one response out, run on a GenerativeBackend's current weights.

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
        return {"backend": self.backend.init_kwargs(), "patch": self.backend.patch_dir}

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
