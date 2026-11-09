"""A `dspy.LM` for a decision backend's local server.

It reaches the server through DSPy's TypeSafe provider exactly as it would reach
TypeSafe's hosted Jev. It also records every call for the forward tool, and
saves the backend's settings and patch, so a saved program reloads with its edit.
"""

from __future__ import annotations

import importlib
import json

import dspy
from dspy.clients.base_lm import LM_CLASS_STATE_KEY

from dspy_wsp.backends.base import LMCall

TIMEOUT_SECONDS = 600.0


class DecisionLM(dspy.LM):
    """`dspy.LM("typesafe:...")` for one local decision backend."""

    @classmethod
    def for_backend(cls, backend, cache: bool = False) -> DecisionLM:
        client = cls(f"typesafe:{backend.model_id}", api_key="local", api_base=backend._server.url, cache=cache, timeout=TIMEOUT_SECONDS)
        client._backend = backend
        return client

    def _record(self, request, response) -> None:
        state = next((part.value for part in request.messages[0].parts if hasattr(part, "value")), request.messages[0].text)
        questions = (request.config.extensions or {}).get("questions", {})
        prompt = json.dumps({"state": state, "questions": questions}, ensure_ascii=False, default=str)
        answers = response.provider_data["typesafe"]["answers"]
        self._backend._record(LMCall(prompt=prompt, completion=json.dumps(answers), finish_reason="stop",
                                     input_tokens=int(response.usage.input_tokens or 0), output_tokens=0))

    def __call__(self, prompt=None, *, cache=None, rollout_id=None, **options):
        # The cache key includes the weights, so a cached answer never outlives an edit.
        rollout_id = self._backend.weights_fingerprint if rollout_id is None else rollout_id
        response = super().__call__(prompt, cache=cache, rollout_id=rollout_id, **options)
        self._record(prompt, response)
        return response

    async def acall(self, prompt=None, *, cache=None, rollout_id=None, **options):
        rollout_id = self._backend.weights_fingerprint if rollout_id is None else rollout_id
        response = await super().acall(prompt, cache=cache, rollout_id=rollout_id, **options)
        self._record(prompt, response)
        return response

    def dump_state(self):
        b = self._backend
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "backend_class": f"{type(b).__module__}.{type(b).__qualname__}",
            "backend": b.init_kwargs(),
            "patch": b.patch_dir,
        }

    @classmethod
    def load_state(cls, state, **_):
        from dspy_wsp.backends.base import LOADED_BACKENDS

        state = {k: v for k, v in state.items() if k != LM_CLASS_STATE_KEY}
        key = json.dumps(state, sort_keys=True, default=str)
        backend = LOADED_BACKENDS.get(key)
        if backend is None:
            module, _, name = state["backend_class"].rpartition(".")
            backend = getattr(importlib.import_module(module), name)(**state["backend"])
            if state.get("patch"):
                from dspy_wsp.patch import Patch

                patch = Patch.load(state["patch"])
                backend.set_weights({n: t.float().numpy() for n, t in patch.edited_tensors(backend.reader).items()})
                backend.patch_dir = state["patch"]
            LOADED_BACKENDS[key] = backend
        return backend.lm()
