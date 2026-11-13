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


def _d1_lm(read: Read) -> bool:
    config = read("config.json") or {}
    return (config.get("model_type") == "lfm3_vl" and (config.get("text_config") or {}).get("model_type") == "lfm3"
            and config.get("d1_attention") == "set")


def _clef(read: Read) -> bool:
    return (read("config.json") or {}).get("model_type") == "qwen3_5" and "routing_layers" in (read("joint_head_config.json") or {})


def _strands(read: Read) -> bool:
    return (read("strands_decider_config.json") or {}).get("head_type") == "pointer"


FAMILIES: list[Family] = [
    Family("d1_omni", "dspy_wsp.backends.decision.D1Backend", _d1_omni,
           "config.json with model_type 'd1_omni'",
           {"d1": ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674")}),
    Family("nanojev", "dspy_wsp.backends.decision.NanoJevBackend", _nanojev,
           "config.json with set_head, and backbone_config/config.json with model_type 'qwen3'",
           {"nanojev": ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a")}),
    Family("d1_lm", "dspy_wsp.backends.decision.D1LMBackend", _d1_lm,
           "config.json with model_type 'lfm3_vl' and d1_attention 'set'",
           {"d1-2b": ("LiquidAI/d1-2B", "652f920e544ea37f81e4b0ce89b81d4ed0da90b1")}),
    Family("clef", "dspy_wsp.backends.decision.ClefBackend", _clef,
           "config.json with model_type 'qwen3_5', and joint_head_config.json",
           {"clef-flash": ("Cloudflare/clef-flash", "22b75507981345858dd152fbd9a6f8c0c4068b0a")}),
    Family("strands", "dspy_wsp.backends.decision.StrandsBackend", _strands,
           "strands_decider_config.json with head_type 'pointer'",
           {"strands-2b": ("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "216d1ccb46a9e9b036eb9187b248db98480f77d4")}),
]


def backend_class(family: Family) -> type:
    module, name = family.backend.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)
