"""DSPy's TypeSafe client, pointed at a decision backend's local server.

It calls the server through the TypeSafe SDK exactly as it would call TypeSafe's
hosted Jev. It also records every call for the forward tool, and saves the
backend's settings and patch, so a saved program reloads with its edit.
"""

from __future__ import annotations

import importlib
import json
import threading

from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM
from dspy.clients.typesafe import TypeSafe

from dspy_wsp.backends.base import LMCall

TIMEOUT_SECONDS = 600.0
_local = threading.local()


class DecisionTypeSafe(TypeSafe, BaseLM):
    """`dspy.experimental.TypeSafe` for one local decision backend.

    It subclasses BaseLM only so DSPy will reload it from a saved program; it
    behaves as a TypeSafe decision client.
    """

    @classmethod
    def for_backend(cls, backend) -> DecisionTypeSafe:
        client = cls(model=backend.model_id, api_key="local", base_url=backend._server.url, cache=False, timeout=TIMEOUT_SECONDS)
        client._backend = backend
        return client

    def copy(self, **kwargs):
        settings = {"model": self.model, "api_key": self.api_key, "base_url": self.base_url, "cache": self.cache,
                    "timeout": self.timeout, "callbacks": self.callbacks, **kwargs}
        clone = type(self)(**settings)
        clone._backend = self._backend
        return clone

    def _request(self, state, questions):
        # The cache key includes the weights, so a cached answer never outlives an edit.
        return {**super()._request(state, questions), "weights": self._backend.weights_fingerprint}

    def _finish(self, request, response, cache_hit):
        _local.usage = response.get("usage", {})
        return super()._finish(request, response, cache_hit)

    def _record(self, state, questions, answers) -> None:
        usage = getattr(_local, "usage", {}) or {}
        prompt = json.dumps({"state": state, "questions": questions}, ensure_ascii=False, default=str)
        self._backend._record(LMCall(prompt=prompt, completion=json.dumps(answers), finish_reason="stop",
                                     input_tokens=int(usage.get("prompt_tokens") or 0), output_tokens=0))

    def __call__(self, state, questions):
        answers = super().__call__(state, questions)
        self._record(state, questions, answers)
        return answers

    async def acall(self, state, questions):
        answers = await super().acall(state, questions)
        self._record(state, questions, answers)
        return answers

    def dump_state(self):
        b = self._backend
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "backend_class": f"{type(b).__module__}.{type(b).__qualname__}",
            "backend": b._init_kwargs,
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
