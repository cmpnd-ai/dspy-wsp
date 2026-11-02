"""The writer's edits and the rules they must follow.

Edits live in `<work>/edited/` as one float32 `.npy` per tensor, named after the
tensor. The kernel's `save_edit` and the host both run `check_edit`, so an edit
written around `save_edit` is still refused before it reaches the model.

This module must not import DSPy: the writer's kernel imports it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

# Out of scope, following the paper's Experiment 8: the input embedding, the
# output layer and the final norm. Patterns are regular expressions matched
# against checkpoint tensor names.
DEFAULT_PROTECTED: tuple[str, ...] = (
    r"(^|\.)embed_tokens\.weight$",
    r"(^|\.)lm_head\.weight$",
    r"(^|\.)wte\.weight$",
    r"(^|\.)norm\.weight$",
    r"(^|\.)ln_f\.(weight|bias)$",
)


class EditError(ValueError):
    """An edit the rules refuse."""


def is_protected(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def check_edit(name: str, new: np.ndarray, original: np.ndarray, protected: Iterable[str] = DEFAULT_PROTECTED) -> dict:
    """Validate one edited tensor against its original and summarize the change."""
    if is_protected(name, protected):
        raise EditError(f"{name} is out of scope: edits to the embeddings, the output layer and the final norm are refused")
    new = np.asarray(new)
    if new.shape != original.shape:
        raise EditError(f"{name}: shape {new.shape} does not match the original {original.shape}")
    if not np.isfinite(new).all():
        raise EditError(f"{name}: the edit contains NaN or infinite values")
    o = original.astype(np.float64).ravel()
    e = new.astype(np.float64).ravel()
    diff = e - o
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm == 0.0:
        raise EditError(f"{name}: the edit equals the original; use drop_edit({name!r}) to remove an edit")
    oo = float(o @ o)
    if oo > 0:
        c = float(e @ o) / oo
        if float(np.linalg.norm(e - c * o)) <= 1e-6 * max(float(np.linalg.norm(e)), 1e-30):
            raise EditError(f"{name}: the edit only multiplies the whole tensor by {c:.6g}; rescaling a whole tensor is out of scope")
    return {
        "name": name,
        "changed_entries": int(np.count_nonzero(diff)),
        "total_entries": int(e.size),
        "relative_change": diff_norm / max(oo**0.5, 1e-30),
    }


def edit_path(edit_dir: str | Path, name: str) -> Path:
    return Path(edit_dir) / f"{name}.npy"


def save_array(edit_dir: str | Path, name: str, arr: np.ndarray) -> Path:
    """Write atomically, so a crash never leaves half an edit behind."""
    path = edit_path(edit_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, path)
    return path


class EditStore:
    """Host-side view of the writer's edit folder."""

    def __init__(self, edit_dir: str | Path, reader, protected: Iterable[str] = DEFAULT_PROTECTED):
        self.dir = Path(edit_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reader = reader
        self.protected = tuple(protected)

    def names(self) -> list[str]:
        return sorted(p.name[: -len(".npy")] for p in self.dir.glob("*.npy"))

    def load_all(self) -> dict[str, np.ndarray]:
        """Every current edit, checked against the rules. Raises EditError on the first bad one."""
        edits: dict[str, np.ndarray] = {}
        for name in self.names():
            if name not in self.reader:
                raise EditError(f"{name}.npy in the edit folder does not name a tensor of the model")
            arr = np.load(edit_path(self.dir, name))
            check_edit(name, arr, self.reader.get_f32(name), self.protected)
            edits[name] = arr.astype(np.float32, copy=False)
        return edits

    def describe(self) -> str:
        names = self.names()
        if not names:
            return "no edits (the model is unchanged)"
        lines = []
        for name in names:
            arr = np.load(edit_path(self.dir, name))
            try:
                s = check_edit(name, arr, self.reader.get_f32(name), self.protected)
                lines.append(f"{name}: {s['changed_entries']:,} of {s['total_entries']:,} entries changed, relative change {s['relative_change']:.4f}")
            except EditError as e:
                lines.append(f"{name}: INVALID ({e})")
        return "\n".join(lines)
