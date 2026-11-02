"""Weight patches: the output of a WeightSpace run.

A patch stores only what changed, plus the sha256 of each base tensor and each
edited tensor (in the model's stored dtype). Applying a patch checks both
hashes, so anyone can rebuild the evaluated weights bit for bit from the public
base model, or learn that their base model differs.

On disk a patch is a folder with `manifest.json` and `edit.npz`, the same shape
as the edits released with the paper's Experiment 8.
"""

from __future__ import annotations

import datetime
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dspy_wsp.tensors import TensorReader, bits_to_tensor, dtype_from_name, dtype_name, sha256_tensor, tensor_bits

FORMAT_VERSION = 1


class PatchError(ValueError):
    """The patch does not match the weights it is applied to."""


@dataclass
class TensorEdit:
    name: str
    shape: tuple[int, ...]
    dtype: str
    form: str  # "sparse" (changed entries only) or "dense" (every entry)
    values: np.ndarray  # raw bits in the stored dtype
    index: np.ndarray | None  # flat indices for "sparse"
    base_sha256: str
    edited_sha256: str
    changed_entries: int
    relative_change: float

    def rebuild(self, base: torch.Tensor) -> torch.Tensor:
        """The edited tensor, checked against both hashes."""
        if tuple(base.shape) != tuple(self.shape) or dtype_name(base.dtype) != self.dtype:
            raise PatchError(f"{self.name}: base is {dtype_name(base.dtype)}{tuple(base.shape)}, patch expects {self.dtype}{self.shape}")
        if sha256_tensor(base) != self.base_sha256:
            raise PatchError(f"{self.name}: the base tensor differs from the one this patch was made on")
        if self.form == "dense":
            bits = self.values
        else:
            bits = tensor_bits(base).reshape(-1).copy()
            bits[self.index] = self.values
        edited = bits_to_tensor(bits, dtype_from_name(self.dtype), self.shape)
        if sha256_tensor(edited) != self.edited_sha256:
            raise PatchError(f"{self.name}: rebuilt tensor does not match the recorded hash")
        return edited


@dataclass
class Patch:
    tensors: list[TensorEdit] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- building -----------------------------------------------------------------
    @classmethod
    def from_edits(cls, reader: TensorReader, edits: Mapping[str, np.ndarray], meta: dict | None = None, dense_threshold: float = 0.25) -> "Patch":
        """Turn float32 edits into a patch in the model's stored dtype.

        Entries that round back to the base value in the stored dtype are not
        changes and are not stored.
        """
        out: list[TensorEdit] = []
        for name in sorted(edits):
            base = reader.get(name)
            edited = torch.from_numpy(np.asarray(edits[name], dtype=np.float32)).reshape(base.shape).to(base.dtype)
            bb, eb = tensor_bits(base).reshape(-1), tensor_bits(edited).reshape(-1)
            changed = np.flatnonzero(bb != eb)
            if changed.size == 0:
                continue
            dense = changed.size > dense_threshold * bb.size
            bf, ef = base.float(), edited.float()
            out.append(
                TensorEdit(
                    name=name,
                    shape=tuple(base.shape),
                    dtype=dtype_name(base.dtype),
                    form="dense" if dense else "sparse",
                    values=eb.copy() if dense else eb[changed].copy(),
                    index=None if dense else changed.astype(np.int64),
                    base_sha256=sha256_tensor(base),
                    edited_sha256=sha256_tensor(edited),
                    changed_entries=int(changed.size),
                    relative_change=float((ef - bf).norm() / bf.norm().clamp_min(1e-30)),
                )
            )
        m = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        m.update(meta or {})
        return cls(out, m)

    # --- disk -----------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        rows = []
        for i, t in enumerate(self.tensors):
            arrays[f"t{i}_val"] = t.values
            if t.index is not None:
                arrays[f"t{i}_idx"] = t.index
            rows.append(
                {
                    "name": t.name,
                    "shape": list(t.shape),
                    "dtype": t.dtype,
                    "form": t.form,
                    "changed_entries": t.changed_entries,
                    "relative_change": t.relative_change,
                    "base_sha256": t.base_sha256,
                    "edited_sha256": t.edited_sha256,
                }
            )
        np.savez_compressed(p / "edit.npz", **arrays)
        manifest = {
            "format": "dspy-wsp-patch",
            "version": FORMAT_VERSION,
            "meta": self.meta,
            "changed_tensors": len(rows),
            "changed_entries_total": sum(r["changed_entries"] for r in rows),
            "tensors": rows,
        }
        (p / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Patch":
        p = Path(path)
        man = json.loads((p / "manifest.json").read_text())
        if man.get("format") != "dspy-wsp-patch":
            raise PatchError(f"{p} is not a dspy-wsp patch")
        tensors = []
        with np.load(p / "edit.npz") as z:
            for i, r in enumerate(man["tensors"]):
                tensors.append(
                    TensorEdit(
                        name=r["name"],
                        shape=tuple(r["shape"]),
                        dtype=r["dtype"],
                        form=r["form"],
                        values=z[f"t{i}_val"],
                        index=z[f"t{i}_idx"] if f"t{i}_idx" in z.files else None,
                        base_sha256=r["base_sha256"],
                        edited_sha256=r["edited_sha256"],
                        changed_entries=r["changed_entries"],
                        relative_change=r["relative_change"],
                    )
                )
        return cls(tensors, man.get("meta", {}))

    # --- applying -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tensors]

    def edited_tensors(self, reader: TensorReader) -> dict[str, torch.Tensor]:
        """Rebuild every edited tensor from the base weights in `reader`."""
        return {t.name: t.rebuild(reader.get(t.name)) for t in self.tensors}

    def apply_to_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for t in self.tensors:
            if t.name not in state:
                raise PatchError(f"{t.name} is not in the state dict")
            state[t.name] = t.rebuild(state[t.name])

    def write_checkpoint(self, base_dir: str | Path, out_dir: str | Path) -> Path:
        """Copy a model folder, replacing the edited tensors.

        The result loads anywhere the base model loads (transformers, vLLM,
        SGLang), so a DSPy program can point a regular `dspy.LM` at it.
        """
        from safetensors.torch import load_file, save_file

        base, out = Path(base_dir), Path(out_dir)
        if out.resolve() == base.resolve():
            raise PatchError("out_dir must differ from base_dir")
        out.mkdir(parents=True, exist_ok=True)
        reader = TensorReader(base)
        by_file: dict[str, list[TensorEdit]] = {}
        for t in self.tensors:
            if t.name not in reader:
                raise PatchError(f"{t.name} is not in {base}")
            by_file.setdefault(reader.index[t.name], []).append(t)
        for f in sorted(base.iterdir()):
            if f.is_dir():
                continue
            if f.name in by_file:
                from safetensors import safe_open

                with safe_open(str(f), framework="pt") as h:
                    metadata = h.metadata() or {"format": "pt"}
                tensors = load_file(str(f))
                for t in by_file[f.name]:
                    tensors[t.name] = t.rebuild(tensors[t.name])
                save_file(tensors, str(out / f.name), metadata=metadata)
            else:
                shutil.copy2(f, out / f.name)
        (out / "wsp_patch.json").write_text((json.dumps({"meta": self.meta, "tensors": self.names}, indent=1, default=str)))
        return out

    def summary(self) -> str:
        if not self.tensors:
            return "empty patch (no weights changed)"
        total = sum(t.changed_entries for t in self.tensors)
        lines = [f"{len(self.tensors)} tensors, {total:,} entries changed"]
        for t in self.tensors:
            lines.append(f"  {t.name} {t.dtype}{list(t.shape)}: {t.changed_entries:,} entries ({t.form}), relative change {t.relative_change:.4f}")
        return "\n".join(lines)
