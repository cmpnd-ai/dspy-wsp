"""Read a model's original tensors straight from its safetensors files.

The writer's kernel, the edit checks and the patch format all compare against
these originals, so nothing here loads the model itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_INT_VIEW = {1: (torch.uint8, np.uint8), 2: (torch.int16, np.uint16), 4: (torch.int32, np.uint32), 8: (torch.int64, np.uint64)}
_SIGNED = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}


def build_index(weights_dir: str | Path) -> dict[str, str]:
    """Map every tensor name to the safetensors file (relative name) that holds it."""
    d = Path(weights_dir)
    index_file = d / "model.safetensors.index.json"
    if index_file.exists():
        return dict(json.loads(index_file.read_text())["weight_map"])
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {d}")
    index: dict[str, str] = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                index[k] = f.name
    return index


def tensor_bits(t: torch.Tensor) -> np.ndarray:
    """The tensor's raw bits as an unsigned-integer numpy array (exact, dtype-agnostic)."""
    t = t.detach().contiguous().cpu()
    tview, nview = _INT_VIEW[t.element_size()]
    return t.view(tview).numpy().view(nview)


def bits_to_tensor(bits: np.ndarray, dtype: torch.dtype, shape) -> torch.Tensor:
    """Inverse of tensor_bits."""
    size = torch.empty((), dtype=dtype).element_size()
    signed = np.ascontiguousarray(bits).view(_SIGNED[size]).copy()
    return torch.from_numpy(signed).view(dtype).reshape(tuple(shape))


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bits(t).tobytes()).hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def dtype_from_name(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r}")
    return dt


class TensorReader:
    """Lazy, read-only access to the original weights in a model folder."""

    def __init__(self, weights_dir: str | Path):
        self.dir = Path(weights_dir)
        self.index = build_index(self.dir)

    @property
    def names(self) -> list[str]:
        return sorted(self.index)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def _path(self, name: str) -> str:
        if name not in self.index:
            raise KeyError(f"unknown tensor {name!r}")
        return str(self.dir / self.index[name])

    def shape(self, name: str) -> tuple[int, ...]:
        with safe_open(self._path(name), framework="pt") as f:
            return tuple(f.get_slice(name).get_shape())

    def shapes(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, tuple[int, ...]] = {}
        by_file: dict[str, list[str]] = {}
        for n, f in self.index.items():
            by_file.setdefault(f, []).append(n)
        for f, names in by_file.items():
            with safe_open(str(self.dir / f), framework="pt") as h:
                for n in names:
                    out[n] = tuple(h.get_slice(n).get_shape())
        return out

    def get(self, name: str) -> torch.Tensor:
        """The original tensor in its stored dtype."""
        with safe_open(self._path(name), framework="pt") as f:
            return f.get_tensor(name)

    def get_f32(self, name: str) -> np.ndarray:
        return self.get(name).float().numpy()
