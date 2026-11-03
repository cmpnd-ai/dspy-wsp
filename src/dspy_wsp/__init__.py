"""Weight-space programming as a DSPy optimizer.

A frontier "writer" LM reads a small model's weights in a Python kernel, runs the
DSPy program through a budgeted forward-pass tool, and edits the weights directly.
No gradients, no training. Method: Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight Space Programming" (2026).

Imports are lazy so the writer's kernel can import `dspy_wsp.edits` without
loading DSPy.
"""

__version__ = "0.1.0"

_EXPORTS = {
    "WeightSpace": "dspy_wsp.optimizer",
    "HFBackend": "dspy_wsp.backends.hf",
    "MLXBackend": "dspy_wsp.backends.mlx",
    "WeightBackend": "dspy_wsp.backends.base",
    "Patch": "dspy_wsp.patch",
    "Limits": "dspy_wsp.forward",
    "Budget": "dspy_wsp.budget",
    "DEFAULT_PROTECTED": "dspy_wsp.edits",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'dspy_wsp' has no attribute {name!r}")
