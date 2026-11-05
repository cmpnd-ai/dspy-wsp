"""The decision-model families dspy-wsp can load, one folder each.

Each folder holds the family's input encoding (`encoding.py`, Python and NumPy
only) and its network in each framework (`torch.py`, `mlx.py`). Its backend
lives in `dspy_wsp.backends.decision`. A family is recognized from its
checkpoint's JSON config files.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

Read = Callable[[str], "dict | None"]  # a checkpoint's JSON file by name, or None


@dataclass(frozen=True)
class Family:
    name: str
    backend: str  # dotted path of the backend class
    matches: Callable[[Read], bool]
    hint: str  # how its config is recognized, for error messages
    aliases: dict[str, tuple[str, str]] = field(default_factory=dict)  # alias: (repo, pinned revision)


def _d1_omni(read: Read) -> bool:
    return (read("config.json") or {}).get("model_type") == "d1_omni"


def _nanojev(read: Read) -> bool:
    return "set_head" in (read("config.json") or {}) and (read("backbone_config/config.json") or {}).get("model_type") == "qwen3"


FAMILIES: list[Family] = [
    Family("d1_omni", "dspy_wsp.backends.decision.D1Backend", _d1_omni,
           "config.json with model_type 'd1_omni'",
           {"d1": ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674")}),
    Family("nanojev", "dspy_wsp.backends.decision.NanoJevBackend", _nanojev,
           "config.json with set_head, and backbone_config/config.json with model_type 'qwen3'",
           {"nanojev": ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a")}),
]


def backend_class(family: Family) -> type:
    module, name = family.backend.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)
