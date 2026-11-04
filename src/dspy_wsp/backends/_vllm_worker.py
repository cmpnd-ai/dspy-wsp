"""Code that runs inside vLLM's worker processes.

`VLLMBackend` names `WeightSwapExtension` as vLLM's `worker_extension_cls`, so
vLLM mixes its methods into every GPU worker. The backend calls them through
`collective_rpc`. This module imports nothing from vLLM at load time, so the
weight stream can be tested without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch


def checkpoint_with_edits(weights_dir: str, edits_path: str | None) -> Iterator[tuple[str, torch.Tensor]]:
    """Every tensor of the original checkpoint, with edited tensors in place of their originals.

    vLLM rebuilds each layer it reloads from scratch, so a reload must deliver every
    tensor of a fused layer (q, k and v together, for example). Streaming the whole
    checkpoint keeps that true for any architecture.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    from dspy_wsp.tensors import build_index

    edits = load_file(edits_path) if edits_path else {}
    index = build_index(weights_dir)
    by_file: dict[str, list[str]] = {}
    for name, file in index.items():
        by_file.setdefault(file, []).append(name)
    missing = sorted(set(edits) - set(index))
    if missing:
        raise KeyError(f"edited tensors not in the checkpoint: {missing[:5]}")
    for file in sorted(by_file):
        with safe_open(str(Path(weights_dir) / file), framework="pt") as h:
            for name in sorted(by_file[file]):
                yield name, edits[name] if name in edits else h.get_tensor(name)


class WeightSwapExtension:
    """Methods vLLM adds to its GPU worker."""

    def wsp_load_weights(self, weights_dir: str, edits_path: str | None) -> None:
        self.reload_weights(weights_iterator=checkpoint_with_edits(weights_dir, edits_path), is_checkpoint_format=True)
        torch.cuda.synchronize()
