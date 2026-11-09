"""Weight-space programming as a DSPy optimizer.

A frontier "writer" LM reads a small model's weights in a Python kernel, runs the
DSPy program through a budgeted forward-pass tool, and edits the weights directly.
No gradients, no training. Method: Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight Space Programming" (2026).

Imports are lazy so the writer's kernel can import `dspy_wsp.edits` without
loading DSPy.
"""

_EXPORTS = {
    "WeightSpace": "dspy_wsp.optimizer",
    "GEPASearch": "dspy_wsp.search",
    "HFBackend": "dspy_wsp.backends.hf",
    "MLXBackend": "dspy_wsp.backends.mlx",
    "VLLMBackend": "dspy_wsp.backends.vllm",
    "D1Backend": "dspy_wsp.backends.decision",
    "NanoJevBackend": "dspy_wsp.backends.decision",
    "D1LMBackend": "dspy_wsp.backends.decision",
    "load_decision_backend": "dspy_wsp.decision.loading",
    "check_decision_model": "dspy_wsp.decision.loading",
    "WeightBackend": "dspy_wsp.backends.base",
    "GenerativeBackend": "dspy_wsp.backends.base",
    "DecisionBackend": "dspy_wsp.backends.decision",
    "Patch": "dspy_wsp.patch",
    "Limits": "dspy_wsp.forward",
    "Budget": "dspy_wsp.budget",
    "DEFAULT_PROTECTED": "dspy_wsp.edits",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("dspy-wsp")
        except PackageNotFoundError:  # a source tree that was never installed
            return "0+unknown"
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'dspy_wsp' has no attribute {name!r}")
