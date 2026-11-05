"""Backends for System One decision models, in PyTorch or MLX.

A decision backend answers typed questions about a state with a probability for
every option; it does not generate text. One worker thread owns the network
(MLX streams belong to the thread that made them). Requests queue to it, and it
answers everything queued with one forward. Weight swaps run on the same
thread between forwards, so every answer comes from one set of weights.

DSPy programs reach the model through `lm()`: a local `/v1/systemone` server and
DSPy's TypeSafe client (`dspy_wsp.decision.client`).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.backends.base import WeightBackend
from dspy_wsp.decision.schema import Question, RequestError, make_answer, parse_question
from dspy_wsp.tensors import TensorReader


class DecisionBackend(WeightBackend):
    """Shared machinery; a subclass supplies the model family's encoding and network."""

    kind = "decision"
    concurrency = 16
    family = ""
    download_patterns: list[str] = []

    def __init__(self, model: str, *, revision: str | None, framework: str | None = None, device: str | None = None, max_length: int | None = None):
        super().__init__()
        self._init_kwargs = {"model": model, "revision": revision, "framework": framework, "device": device, "max_length": max_length}
        self.patch_dir: str | None = None
        path = Path(model)
        if path.is_dir():
            self._dir = path.resolve()
        else:
            from huggingface_hub import snapshot_download

            self._dir = Path(snapshot_download(model, revision=revision, allow_patterns=self.download_patterns))
        self._model_id = model
        self.revision = revision
        self.framework = framework or _default_framework()
        self.device = device
        self.reader = TensorReader(self._dir)
        self._setup(max_length)
        self._client = None
        self._server = None
        self._tasks: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._serve, name=f"wsp-{self.family}", daemon=True)
        self._worker.start()
        self._call(self._load)

    # --- family hooks -------------------------------------------------------------------
    def _setup(self, max_length: int | None) -> None:
        """Read the config and tokenizer (any thread)."""
        raise NotImplementedError

    def _load_network(self):
        raise NotImplementedError

    def _encode(self, state: Any, q: Question):
        """One row for the network, or RequestError if the request does not fit."""
        raise NotImplementedError

    def _probabilities(self, logits: np.ndarray, q: Question) -> list[float]:
        raise NotImplementedError

    def _model_tensor(self, name: str) -> bool:
        """Whether a checkpoint tensor belongs to the loaded network."""
        return True

    # --- worker thread ------------------------------------------------------------------
    def _serve(self) -> None:
        while True:
            items = [self._tasks.get()]
            while True:
                try:
                    items.append(self._tasks.get_nowait())
                except queue.Empty:
                    break
            decides = [it for it in items if it[0] == "decide"]
            if decides:
                self._decide(decides)
            for kind, payload, fut in items:
                if kind == "stop":
                    fut.set_result(None)
                    return
                if kind == "call":
                    try:
                        fut.set_result(payload())
                    except BaseException as e:  # noqa: BLE001 - failures go back to the caller
                        fut.set_exception(e)

    def _decide(self, items) -> None:
        rows, owners = [], []
        for _, (state, questions), fut in items:
            try:
                encoded = [self._encode(state, q) for q in questions.values()]
            except Exception as e:  # noqa: BLE001 - a bad request fails alone
                fut.set_exception(e if isinstance(e, RequestError) else RequestError(str(e)))
                continue
            owners.append((fut, questions, len(rows), len(encoded), sum(len(r[0]) for r in encoded)))
            rows.extend(encoded)
        if not rows:
            return
        try:
            logits = self.net.logits(rows)
        except BaseException as e:  # noqa: BLE001
            for fut, *_ in owners:
                fut.set_exception(e)
            return
        for fut, questions, start, n, tokens in owners:
            answers = {name: make_answer(q, self._probabilities(z, q)) for (name, q), z in zip(questions.items(), logits[start : start + n])}
            fut.set_result({"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})

    def _call(self, fn):
        fut: Future = Future()
        self._tasks.put(("call", fn, fut))
        return fut.result()

    def _load(self) -> None:
        self.net = self._load_network()
        self._params = self._named_params()
        self._param_for = {n: n for n in self.reader.names if n in self._params and self._model_tensor(n)}
        self._original: dict[str, Any] = {}

    # --- framework-specific parameter access --------------------------------------------
    def _named_params(self) -> dict:
        if self.framework == "torch":
            return dict(self.net.named_parameters())
        from mlx.utils import tree_flatten

        return dict(tree_flatten(self.net.parameters()))

    def _apply(self, edits: Mapping[str, np.ndarray]) -> None:
        unknown = [n for n in edits if n not in self._param_for]
        if unknown:
            raise KeyError(f"{unknown[0]} does not map to a parameter of the loaded network")
        self._call(lambda: self._swap(edits))

    def _swap(self, edits) -> None:
        if self.framework == "torch":
            import torch

            with torch.no_grad():
                for name, saved in self._original.items():
                    self._params[name].data.copy_(saved)
                self._original = {}
                for name, arr in edits.items():
                    p = self._params[name]
                    self._original[name] = p.data.detach().clone()
                    p.data.copy_(torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(p.shape).to(p.dtype))
            return
        import mlx.core as mx

        updates = dict(self._original)
        self._original = {}
        current = self._named_params()
        for name, arr in edits.items():
            self._original[name] = updates.get(name, current[name])
            updates[name] = mx.array(np.asarray(arr, dtype=np.float32).reshape(current[name].shape)).astype(current[name].dtype)
        if updates:
            self.net.load_named(updates)
        self._params = self._named_params()

    # --- answering ----------------------------------------------------------------------
    def system_one(self, state: Any, questions: Mapping[str, Any]) -> dict:
        return self.system_one_batch([(state, questions)])[0]

    def system_one_batch(self, requests: Sequence[tuple[Any, Mapping[str, Any]]]) -> list[dict]:
        futures = []
        for state, questions in requests:
            parsed = {name: q if isinstance(q, Question) else parse_question(q) for name, q in questions.items()}
            if not parsed:
                raise RequestError("`questions` must be a non-empty object")
            fut: Future = Future()
            self._tasks.put(("decide", (state, parsed), fut))
            futures.append(fut)
        return [f.result() for f in futures]

    def read(self, requests: Sequence[Mapping[str, Any]], layers: Sequence[int] = ()) -> tuple[dict[str, np.ndarray], dict]:
        """What the model computes for each question of each request, padded to the most options.

        Arrays (one row per question): option_logits and probabilities (rows, k), option_mask (rows, k),
        head_inputs (rows, k, d) and the family's own captures; hidden_mean (rows, len(layers), d) if asked.
        """
        layers = list(layers)
        L = self.dims()["layers"]
        if any(not isinstance(l, int) or not 0 <= l <= L for l in layers):
            raise RequestError(f"layers must be integers from 0 to {L}")
        rows, index = [], []
        for i, req in enumerate(requests):
            for name, q in req["questions"].items():
                q = q if isinstance(q, Question) else parse_question(q)
                rows.append(self._encode(req.get("state", ""), q))
                index.append({"request": i, "question": name, "type": q.type, "options": q.option_keys})
        caps = self._call(lambda: self.net.capture(rows, layers))
        k = max(len(r["options"]) for r in index)
        arrays: dict[str, np.ndarray] = {"option_mask": np.zeros((len(rows), k), bool)}
        for j, (cap, meta) in enumerate(zip(caps, index)):
            n = len(meta["options"])
            arrays["option_mask"][j, :n] = True
            q = rows[j][-1]
            cap = {**cap, "probabilities": np.array(self._probabilities(cap["logits"], q), np.float32)}
            for key, value in cap.items():
                name = "option_logits" if key == "logits" else key
                if name not in arrays:
                    shape = (len(rows), k) + value.shape[1:] if key != "hidden_mean" else (len(rows),) + value.shape
                    arrays[name] = np.full(shape, np.nan, np.float32)
                if key == "hidden_mean":
                    arrays[name][j] = value
                else:
                    arrays[name][j, :n] = value
        meta = {"rows": index, "layers": layers, "conventions": self.read_conventions()}
        return arrays, meta

    def read_conventions(self) -> dict:
        return {
            "rows": "one row per question, in request order; meta.rows names each row's request, question and options",
            "option_mask": "True on a question's real options; padded options are NaN",
            "option_logits": "the head's raw score per option, before calibration",
            "probabilities": "the answer's distribution over the options",
            "hidden_mean": "hidden state after each listed layer (0 = embeddings, l = after block l-1), averaged over the row's tokens",
        }

    # --- WeightBackend surface ----------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def weights_dir(self) -> Path:
        return self._dir

    def editable_names(self) -> list[str]:
        return sorted(self._param_for)

    def default_protected(self) -> tuple[str, ...]:
        return ()

    @property
    def tokenizer_dir(self) -> Path:
        return self._dir

    def render_chat(self, messages, *, add_generation_prompt=False, **kw) -> str:
        raise NotImplementedError("decision models answer typed questions; they have no chat template")

    def teacher_forced(self, texts, **kw):
        raise NotImplementedError("decision models have no text reads; use read()")

    def pooled_activations(self, pairs, **kw):
        raise NotImplementedError("decision models have no text reads; use read()")

    def lm(self):
        if self._client is None:
            from dspy_wsp.decision.client import DecisionTypeSafe
            from dspy_wsp.decision.server import SystemOneServer

            self._server = SystemOneServer(self)
            self._server.start()
            self._client = DecisionTypeSafe.for_backend(self)
        return self._client

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._worker.is_alive():
            fut: Future = Future()
            self._tasks.put(("stop", None, fut))
            fut.result()


def _default_framework() -> str:
    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        return "torch"


class D1Backend(DecisionBackend):
    """Liquid AI's d1 decision models (text only). Defaults to d1-omni-600M at a pinned revision."""

    family = "d1"
    download_patterns = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "LICENSE"]

    def __init__(self, model: str = "LiquidAI/d1-omni-600M", *, revision: str | None = "02b55d7076f15129e59ab3f94783f32c4b088674", framework: str | None = None, device: str | None = None, max_length: int | None = None):
        super().__init__(model, revision=revision, framework=framework, device=device, max_length=max_length)

    def _setup(self, max_length):
        from dspy_wsp.decision import d1

        self.config = d1.load_config(self._dir)
        self.tok = d1.Tokenizer(self._dir)
        self.max_length = int(max_length or self.config["max_length"])

    def _load_network(self):
        if self.framework == "torch":
            from dspy_wsp.decision.d1_torch import D1Torch

            return D1Torch.from_dir(self._dir, device=self.device or "cpu")
        from dspy_wsp.decision.d1_mlx import D1MLX

        return D1MLX.from_dir(self._dir)

    def _encode(self, state, q):
        from dspy_wsp.decision import d1

        try:
            ids, markers = d1.encode(self.tok, state, q, self.max_length)
        except ValueError as e:
            raise RequestError(str(e)) from None
        return ids, markers, q

    def _probabilities(self, logits, q):
        from dspy_wsp.decision import d1

        return d1.probabilities(logits, q, self.config["temperatures"])

    def _model_tensor(self, name):
        return name.startswith(("encoder.", "head."))

    def default_protected(self):
        return (r"^encoder\.embed_tokens\.weight$", r"^encoder\.embedding_norm\.weight$")

    protected_description = "the tensors marked (out of scope) below (the token embedding and the encoder's final norm)"

    def num_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))

    def dims(self) -> dict:
        t = self.config["text_config"]
        return {"layers": t["num_hidden_layers"], "hidden": t["hidden_size"], "intermediate": t["intermediate_size"],
                "heads": t["num_attention_heads"], "kv_heads": t["num_key_value_heads"], "vocab": t["vocab_size"]}

    def describe(self) -> str:
        t = self.config["text_config"]
        n = sum(int(np.prod(self.reader.shape(k))) for k in self._param_for)
        kinds = t["layer_types"]
        attn = [i for i, k in enumerate(kinds) if k == "full_attention"]
        return (
            f"{self.model_id} (d1 decision model, text path): a bidirectional LFM2 encoder with {len(kinds)} blocks, width "
            f"{t['hidden_size']}; blocks {attn} use attention and the others a centred 3-tap short convolution. "
            f"Each question is read with the state as one sequence: <bos> <state> state <q> instructions, then for each option "
            f"<opt> <mask> option text </opt>, then <decide>. The decision head (`head.*`) adds a question-type embedding "
            f"(`head.type_emb`, rows choice, score, noul), runs {self.config['head_layers']} pre-norm transformer layers "
            f"(`head.head.layers.*`) and scores the hidden state at each option's <mask> with an MLP (`head.scorer`: "
            f"LayerNorm, Linear, GELU, Linear to one logit). Logits are divided by a calibrated temperature per question type "
            f"and option count, then softmaxed per question. A noul's options are [false, true]. {n:,} parameters in float32."
        )

    def read_conventions(self) -> dict:
        return {**super().read_conventions(),
                "head_inputs": "the scorer's input at each option's <mask> (after the head's transformer layers)",
                "trunk_markers": "the trunk's normed output at each option's <mask> (the head's input there)"}
