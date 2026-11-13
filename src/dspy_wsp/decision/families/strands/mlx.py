"""The Strands decider network in MLX: a Qwen3.5 backbone and the pointer head.

A released checkpoint holds a LoRA adapter and a head; its backbone weights are
the base model's. `fold` writes one folder that holds the whole network: the
base weights with the adapter folded in, W + (alpha / r) B A, and the head.
`StrandsMLX` loads that folder, so every weight is a tensor on disk that an
edit can name.

The backbone is mlx-lm's Qwen3.5 text model. Its tensors are stored as mlx-lm
holds them, under `backbone.` and the module's own path, e.g.
`backbone.layers.0.mlp.gate_proj.weight`. The head (`PointerHead` in the
`strands-decider` package) runs in float32, under the names of `head.safetensors`:

    decide = backbone output at the prompt's last token
    option = backbone output at the last token of the option's line
    logit  = k(norm(option)) . q(norm(decide)) / sqrt(dim)

The backbone's outputs are cached per prompt. The cache for the unedited backbone
is kept; an edited backbone gets its own, which is dropped when the edit changes.

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision.families.strands.encoding import Row

CACHE_BYTES = 2 << 30  # backbone outputs kept for reuse, per cache
BATCH_ROWS = 16
BATCH_TOKENS = 16_384  # rows x padded length of one backbone forward
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
BACKBONE = "backbone."
CONFIG = "strands_decider_config.json"
COPIED = ["head.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "provenance.json", "LICENSE.md"]


def merge_lora(model, adapter_dir: Path) -> int:
    """Fold a PEFT LoRA adapter into the weights of mlx-lm's `model`. Returns the number of weights merged."""
    from mlx.utils import tree_flatten

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    unsupported = [key for key in ("fan_in_fan_out", "rank_pattern", "alpha_pattern", "use_dora") if cfg.get(key)]
    if unsupported:
        raise ValueError(f"{adapter_dir}: cannot merge an adapter with {unsupported}")
    adapter = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    stems = {name.removesuffix(".lora_A.weight") for name in adapter if name.endswith(".lora_A.weight")}
    extra = set(adapter) - {stem + suffix for stem in stems for suffix in (".lora_A.weight", ".lora_B.weight")}
    if extra:
        raise ValueError(f"{adapter_dir}: only lora_A and lora_B weight pairs can be merged, not {sorted(extra)[:3]}")
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    params = dict(tree_flatten(model.parameters()))
    # Metal's float32 matmul loses precision; the merge runs once, on the CPU.
    with mx.stream(mx.cpu):
        for stem in sorted(stems):
            target = "language_model.model." + stem.removeprefix("base_model.model.") + ".weight"
            if target not in params:
                raise ValueError(f"adapter tensor {stem} has no weight {target} in the backbone")
            base = params[target]
            delta = adapter[stem + ".lora_B.weight"].astype(mx.float32) @ adapter[stem + ".lora_A.weight"].astype(mx.float32)
            weight = (base.astype(mx.float32) + scale * delta).astype(base.dtype)
            mx.eval(weight)
            model.load_weights([(target, weight)], strict=False)
    return len(stems)


def fold(checkpoint: Path, base: Path, out: Path, source: dict | None = None) -> Path:
    """Write `out`: the checkpoint's whole network in one folder, with the adapter folded into the base weights."""
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    config = json.loads((checkpoint / CONFIG).read_text())
    base_config = json.loads((base / "config.json").read_text())
    model = Model(ModelArgs.from_dict(base_config))
    weights: dict = {}
    for f in sorted(base.glob("model*.safetensors")):
        weights.update(mx.load(str(f)))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    dtype = DTYPES[config.get("torch_dtype", "bfloat16")]
    keep = model.cast_predicate
    cast = [(path, value.astype(dtype)) for path, value in tree_flatten(model.parameters())
            if mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype and keep(path)]
    if cast:
        model.load_weights(cast, strict=False)
    merged = merge_lora(model, checkpoint / "lora") if config.get("use_lora") else 0
    backbone = {BACKBONE + name: value for name, value in tree_flatten(model.language_model.model.parameters())}
    mx.eval(backbone)

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    mx.save_safetensors(str(partial / "model.safetensors"), backbone, metadata={"format": "pt"})
    for name in COPIED:
        if (checkpoint / name).is_file():
            shutil.copy2(checkpoint / name, partial / name)
    (partial / "config.json").write_text(json.dumps(base_config, indent=1))
    folded = {**config, "use_lora": False, "folded": {**(source or {}), "base_model": config["base_model"], "lora_weights_merged": merged}}
    (partial / CONFIG).write_text(json.dumps(folded, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    del model, weights, backbone
    mx.clear_cache()
    return out


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, dim)
        self.k = nn.Linear(hidden_size, dim)

    def score(self, decide, options):
        """Logits (options,) from `decide` (hidden,) and `options` (options, hidden), in float32."""
        return (self.k(self.norm(options)) @ self.q(self.norm(decide))) * self.q.weight.shape[0] ** -0.5


class StrandsMLX(PointerHead):
    @classmethod
    def from_dir(cls, folder: str | Path) -> StrandsMLX:
        """Load a folded checkpoint folder."""
        from mlx_lm.models.qwen3_5 import Model, ModelArgs

        folder = Path(folder)
        config = json.loads((folder / CONFIG).read_text())
        backbone = Model(ModelArgs.from_dict(json.loads((folder / "config.json").read_text()))).language_model.model
        weights = mx.load(str(folder / "model.safetensors"))
        backbone.load_weights([(name.removeprefix(BACKBONE), value) for name, value in weights.items()], strict=True)
        backbone.eval()
        net = cls(backbone.embed_tokens.weight.shape[1], config["pointer_dim"])
        net.backbone = backbone
        head = mx.load(str(folder / "head.safetensors"))
        if set(head) != {"norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"}:
            raise ValueError(f"{folder / 'head.safetensors'} is not a pointer head: {sorted(head)}")
        net._stored = {k: v.dtype for k, v in head.items()}
        net.load_named(head)
        net._caches = {"original": OrderedDict()}
        net._key = "original"
        return net

    def load_named(self, weights: dict, strict: bool = False) -> None:
        # Round a head tensor to the checkpoint's dtype first, so the head computes with the values a patch stores.
        weights = {k: v.astype(self._stored[k]).astype(mx.float32) if k in self._stored else v for k, v in weights.items()}
        self.load_weights(list(weights.items()), strict=False)
        mx.eval(self.parameters())

    def set_backbone_key(self, key: str) -> None:
        """Name the backbone's current weights ("original" when unedited), which selects the output cache."""
        self._caches = {k: v for k, v in self._caches.items() if k in ("original", key)}
        self._caches.setdefault(key, OrderedDict())
        self._key = key

    def picked(self, rows: list[Row]) -> list:
        """Per row, the backbone's output at its last token and at each option's last token: (1 + options, hidden) in float32."""
        cache = self._caches[self._key]
        todo = {row.ids: row for row in rows if row.ids not in cache}
        pending = sorted(todo.values(), key=lambda row: len(row.ids), reverse=True)
        while pending:
            width = len(pending[0].ids)
            batch = pending[: max(1, min(BATCH_ROWS, BATCH_TOKENS // width))]
            pending = pending[len(batch):]
            # Every layer is causal, so right padding changes no real token's output.
            ids = mx.array([list(row.ids) + [0] * (width - len(row.ids)) for row in batch])
            hidden = self.backbone(ids)
            outs = [hidden[i][mx.array([len(row.ids) - 1, *row.options])].astype(mx.float32) for i, row in enumerate(batch)]
            mx.eval(*outs)
            cache.update((row.ids, out) for row, out in zip(batch, outs))
        result = []
        for row in rows:
            cache.move_to_end(row.ids)
            result.append(cache[row.ids])
        size = sum(out.nbytes for out in cache.values())
        while size > CACHE_BYTES and len(cache) > 1:
            size -= cache.popitem(last=False)[1].nbytes
        return result

    def layer_means(self, row: Row, keep: list[int]) -> dict:
        """The state's mean hidden state after each layer in `keep` (0 = embeddings)."""
        from mlx_lm.models.base import create_attention_mask

        h = self.backbone.embed_tokens(mx.array([list(row.ids)]))
        mask = create_attention_mask(h, None)
        end = row.state or len(row.ids)
        kept = {0: h[0, :end].mean(axis=0)} if 0 in keep else {}
        for i, layer in enumerate(self.backbone.layers[: max(keep)]):
            h = layer(h, mask=None if layer.is_linear else mask, cache=None)
            if i + 1 in keep:
                kept[i + 1] = h[0, :end].mean(axis=0)
        mx.eval(*kept.values())
        return kept

    def capture(self, rows: list[Row], layers: list[int]) -> list[dict[str, np.ndarray]]:
        out = []
        for row, picked in zip(rows, self.picked(rows)):
            decide, options = picked[0], picked[1:]
            logits = self.score(decide, options)
            mx.eval(logits)
            rec = {"logits": np.array(logits, dtype=np.float32), "head_inputs": np.array(options, dtype=np.float32),
                   "decide": np.broadcast_to(np.array(decide, dtype=np.float32), options.shape).copy()}
            if layers:
                kept = self.layer_means(row, list(layers))
                rec["hidden_mean"] = np.stack([np.array(kept[l].astype(mx.float32)) for l in layers])
            out.append(rec)
        return out

    def logits(self, rows: list[Row]) -> list[np.ndarray]:
        return [rec["logits"] for rec in self.capture(rows, [])]
