"""The Clef network in MLX: a fixed Qwen3.5 backbone and the joint schema head.

The backbone is mlx-lm's Qwen3.5 text model with the checkpoint's weights, in
the checkpoint's dtype. It is not a parameter of this module: edits reach only
the head, so a prompt's backbone output never changes and is cached.

The head (`JointSchemaHead` in the checkpoint's `joint_schema_model.py`) runs in
float32, and its parameter names equal the tensor names in `joint_head.safetensors`:

    x      = hidden_norm(backbone output)                       one row per token
    memory = memory_projection(x)
    each option's query = option_context_projection(mean of x over the option's tokens)
                        + option_lexical_projection(mean of the output-embedding rows of those tokens)
                        + option_question_projection(mean of x over the question's tokens)
    evidence_layers: every option query reads `memory` by cross-attention
    each question's field = question_projection(question mean) + option_summary_norm(a softmax-weighted
                            sum of its routed options) + global_projection(x at the last token) + type_embedding
    layers: the fields read each other and `memory` (pre-norm transformer decoder layers), then field_norm
    logit  = prior + sigmoid(residual_gate) * (exp(joint_logit_scale) * cosine + residual_scorer(features))

`features` is [field, option, field * option, |field - option|] with option = option_norm(routed option),
`cosine` is the cosine of field and option, and `prior` is exp(prior_logit_scale) times the cosine of the
option's lexical mean and (question mean + x at the last token).

Call it from one thread: MLX streams belong to the thread that made them.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dspy_wsp.decision.families.clef.encoding import Row

CACHE_BYTES = 6 << 30  # backbone outputs kept for reuse
MAX_SCALE = math.log(100.0)


class Attention(nn.Module):
    """`torch.nn.MultiheadAttention` with a packed input projection: `queries` (n, d) read `memory` (s, d)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * d, d))
        self.in_proj_bias = mx.zeros((3 * d,))
        self.out_proj = nn.Linear(d, d)

    def __call__(self, queries, memory):
        d = queries.shape[-1]
        wq, wk, wv = mx.split(self.in_proj_weight, 3, axis=0)
        bq, bk, bv = mx.split(self.in_proj_bias, 3)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, d // self.heads).transpose(0, 2, 1, 3)  # noqa: E731
        q, k, v = split(queries @ wq.T + bq), split(memory @ wk.T + bk), split(memory @ wv.T + bv)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // self.heads) ** -0.5, mask=None)
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(queries.shape[0], d))


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        # Indices 2 and 4 are the reference's dropouts.
        self.feedforward = [nn.Linear(width, feedforward), nn.GELU(), nn.Identity(), nn.Linear(feedforward, width), nn.Identity()]

    def __call__(self, queries, memory):
        queries = queries + self.attention(self.query_norm(queries), self.memory_norm(memory))
        y = self.feedforward_norm(queries)
        for f in self.feedforward:
            y = f(y)
        return queries + y


class DecoderLayer(nn.Module):
    """`torch.nn.TransformerDecoderLayer` with norm_first=True, GELU and no dropout."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.multihead_attn = Attention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x, memory):
        y = self.norm1(x)
        x = x + self.self_attn(y, y)
        x = x + self.multihead_attn(self.norm2(x), memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _unit(x, eps: float):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size: int, width: int, routing_layers: int, layers: int, heads: int, feedforward: int):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [DecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        # Index 2 is the reference's dropout.
        self.residual_scorer = [nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)]
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def score(self, hidden, row: Row, lexical: list) -> list[dict]:
        """Each question's logits and scorer inputs. `hidden` is the backbone's output (tokens, hidden) in float32,
        and `lexical` holds, per question, each option's mean output-embedding row."""
        x = self.hidden_norm(hidden)
        memory = self.memory_projection(x)
        last = x[-1]
        mean = lambda span: x[span[0] : span[1]].mean(axis=0)  # noqa: E731
        questions = mx.stack([mean(f.question) for f in row.fields])
        queries = [
            self.option_context_projection(mx.stack([mean(s) for s in f.options])) + self.option_lexical_projection(lex)
            + self.option_question_projection(questions[i])[None]
            for i, (f, lex) in enumerate(zip(row.fields, lexical))
        ]
        routed = mx.concatenate(queries, axis=0)
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        bounds = np.cumsum([0] + [len(f.options) for f in row.fields])
        options = [routed[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

        base = self.question_projection(questions)
        summaries = []
        for field, opts in zip(base, options):
            weights = mx.softmax(opts @ field / math.sqrt(opts.shape[-1]), axis=0)
            summaries.append((weights[:, None] * opts).sum(axis=0))
        types = mx.array([f.type_id for f in row.fields])
        fields = base + self.option_summary_norm(mx.stack(summaries)) + self.global_projection(last)[None] + self.type_embedding(types)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields)

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, MAX_SCALE))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, MAX_SCALE))
        out = []
        for i, (opts, lex) in enumerate(zip(options, lexical)):
            prior = prior_scale * (_unit(lex, 1e-12) @ _unit(questions[i] + last, 1e-12))
            opts = self.option_norm(opts)
            field = mx.broadcast_to(fields[i][None], opts.shape)
            cosine = (_unit(field, 1e-8) * _unit(opts, 1e-8)).sum(axis=-1)
            features = mx.concatenate([field, opts, field * opts, mx.abs(field - opts)], axis=-1)
            residual = features
            for f in self.residual_scorer:
                residual = f(residual)
            logits = prior + mx.sigmoid(self.residual_gate) * (joint_scale * cosine + residual[:, 0])
            out.append({"logits": logits, "head_inputs": features, "prior": prior, "cosine": cosine})
        return out


class Backbone:
    """mlx-lm's Qwen3.5 text model with the checkpoint's weights. Not an `nn.Module`, so the head's parameters exclude it."""

    def __init__(self, folder: Path, config: dict):
        from mlx_lm.models.qwen3_5 import Model, ModelArgs

        model = Model(ModelArgs.from_dict(config))
        weights: dict = {}
        for f in sorted(folder.glob("model*.safetensors")):
            weights.update(mx.load(str(f)))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        self.text = model.language_model.model
        # The head's lexical features read the output embedding.
        self.output_embedding = model.language_model.lm_head.weight if hasattr(model.language_model, "lm_head") else self.text.embed_tokens.weight
        self._cache: OrderedDict = OrderedDict()
        self._cached_bytes = 0

    def __call__(self, row: Row, keep: list[int]):
        """The final normed hidden states (tokens, hidden), and the state's mean hidden state after each layer in `keep`."""
        if not keep and row.ids in self._cache:
            self._cache.move_to_end(row.ids)
            return self._cache[row.ids], {}
        from mlx_lm.models.base import create_attention_mask

        a, b = row.state if row.state[1] > row.state[0] else (0, len(row.ids))
        h = self.text.embed_tokens(mx.array([list(row.ids)]))
        mask = create_attention_mask(h, None)
        kept = {0: h[0, a:b].mean(axis=0)} if 0 in keep else {}
        for i, layer in enumerate(self.text.layers):
            h = layer(h, mask=None if layer.is_linear else mask, cache=None)
            if i + 1 in keep:
                kept[i + 1] = h[0, a:b].mean(axis=0)
        last = self.text.norm(h)[0]
        mx.eval(last, *kept.values())
        if row.ids not in self._cache:
            self._cache[row.ids] = last
            self._cached_bytes += last.nbytes
            while self._cached_bytes > CACHE_BYTES and len(self._cache) > 1:
                self._cached_bytes -= self._cache.popitem(last=False)[1].nbytes
        return last, kept

    def lexical(self, row: Row) -> list:
        ids = mx.array(list(row.ids))
        return [mx.stack([self.output_embedding[ids[a:b]].astype(mx.float32).mean(axis=0) for a, b in f.options]) for f in row.fields]


class ClefMLX(JointSchemaHead):
    @classmethod
    def from_dir(cls, folder: str | Path) -> ClefMLX:
        folder = Path(folder)
        net = cls(**json.loads((folder / "joint_head_config.json").read_text()))
        weights = mx.load(str(folder / "joint_head.safetensors"))
        net._stored = {k: v.dtype for k, v in weights.items()}
        net.load_named(weights, strict=True)
        net._backbone = Backbone(folder, json.loads((folder / "config.json").read_text()))
        return net

    def load_named(self, weights: dict, strict: bool = False) -> None:
        # Round to the checkpoint's dtype first, so the head computes with the values a patch stores.
        weights = {k: v.astype(self._stored[k]).astype(mx.float32) for k, v in weights.items()}
        self.load_weights(list(weights.items()), strict=strict)
        mx.eval(self.parameters())

    def capture(self, rows: list[Row], layers: list[int]) -> list[list[dict[str, np.ndarray]]]:
        out = []
        for row in rows:
            hidden, kept = self._backbone(row, list(layers))
            recs = self.score(hidden.astype(mx.float32), row, self._backbone.lexical(row))
            mx.eval(*[v for rec in recs for v in rec.values()])
            recs = [{k: np.array(v, dtype=np.float32) for k, v in rec.items()} for rec in recs]
            if layers:
                means = np.stack([np.array(kept[l].astype(mx.float32)) for l in layers])
                for rec in recs:
                    rec["hidden_mean"] = means
            out.append(recs)
        return out

    def logits(self, rows: list[Row]) -> list[list[np.ndarray]]:
        return [[rec["logits"] for rec in recs] for recs in self.capture(rows, [])]
