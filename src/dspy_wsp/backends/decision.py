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
from typing import Any, ClassVar

import numpy as np

from dspy_wsp.backends.base import WeightBackend
from dspy_wsp.decision.schema import Question, RequestError, make_answer, parse_question
from dspy_wsp.tensors import TensorReader


MLX_CACHE_LIMIT = 1 << 30  # bytes of freed MLX buffers kept for reuse


class DecisionBackend(WeightBackend):
    """Shared machinery; a subclass supplies the model family's encoding and network."""

    kind: ClassVar[str] = "decision"
    concurrency = 16
    family = ""
    download_patterns: list[str] = []

    def __init__(self, model: str, *, revision: str | None, framework: str | None = None, device: str | None = None, max_length: int | None = None):
        super().__init__()
        self._init_kwargs = {"model": model, "revision": revision, "framework": framework, "device": device, "max_length": max_length}
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

    def _row_tokens(self, row) -> int:
        return len(row[0])

    def _encode_request(self, state: Any, questions: Mapping[str, Question]) -> list[tuple[Any, list[str]]]:
        """The network's rows for one request, each with the names of the questions it answers, in order."""
        return [(self._encode(state, q), [name]) for name, q in questions.items()]

    def _check_question(self, raw: Any) -> None:
        """Refuse a question the family cannot answer, before it is parsed."""

    def _parse_question(self, q: Any) -> Question:
        self._check_question(q)
        return q if isinstance(q, Question) else parse_question(q)

    def _row_outputs(self, output, n: int) -> list:
        """One item per question from the network's output for one row (`logits` or `capture`)."""
        return [output]

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
                try:
                    self._decide(decides)
                except BaseException as e:  # noqa: BLE001 - fail the waiting callers, keep the worker alive
                    for _, _, fut in decides:
                        if not fut.done():
                            fut.set_exception(e)
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
                encoded = self._encode_request(state, questions)
                tokens = sum(self._row_tokens(row) for row, _ in encoded)
            except Exception as e:  # noqa: BLE001 - a bad request fails alone
                fut.set_exception(e if isinstance(e, RequestError) else RequestError(str(e)))
                continue
            owners.append((fut, questions, len(rows), encoded, tokens))
            rows.extend(row for row, _ in encoded)
        if not rows:
            return
        try:
            logits = self.net.logits(rows)
        except BaseException as e:  # noqa: BLE001
            for fut, *_ in owners:
                fut.set_exception(e)
            return
        for fut, questions, start, encoded, tokens in owners:
            by_name = {}
            for (_, names), out in zip(encoded, logits[start : start + len(encoded)]):
                by_name.update(zip(names, self._row_outputs(out, len(names))))
            answers = {name: make_answer(q, self._probabilities(by_name[name], q)) for name, q in questions.items()}
            fut.set_result({"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})

    def _call(self, fn):
        fut: Future = Future()
        self._tasks.put(("call", fn, fut))
        return fut.result()

    def _load(self) -> None:
        if self.framework == "mlx":
            import mlx.core as mx

            # MLX keeps freed buffers for reuse. Batches of many different shapes grow that cache
            # without bound (32 GB after 30 NanoJev cases), so cap it.
            mx.set_cache_limit(MLX_CACHE_LIMIT)
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

    @property
    def weights_fingerprint(self) -> str:
        """A hash of the current edits ("original" with none); part of every cache key, so cached answers match the weights."""
        return getattr(self, "_fingerprint", "original")

    def _fingerprint_of(self, edits) -> str:
        import hashlib

        if not edits:
            return "original"
        h = hashlib.sha256()
        for name in sorted(edits):
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.asarray(edits[name], dtype=np.float32)).tobytes())
        return h.hexdigest()[:16]

    def _swap(self, edits) -> None:
        self._fingerprint = self._fingerprint_of(edits)
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
            parsed = {name: self._parse_question(q) for name, q in questions.items()}
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
        rows, plan = [], []  # plan: (request, its questions, its encoded rows)
        for i, req in enumerate(requests):
            questions = {name: self._parse_question(q) for name, q in req["questions"].items()}
            encoded = self._encode_request(req.get("state", ""), questions)
            plan.append((i, questions, encoded))
            rows.extend(row for row, _ in encoded)
        caps = self._call(lambda: self.net.capture(rows, layers))
        flat, index, at = [], [], 0
        for i, questions, encoded in plan:
            by_name = {}
            for (_, names), cap in zip(encoded, caps[at : at + len(encoded)]):
                by_name.update(zip(names, self._row_outputs(cap, len(names))))
            at += len(encoded)
            for name, q in questions.items():
                flat.append((by_name[name], q))
                index.append({"request": i, "question": name, "type": q.type, "options": q.option_keys})
        k = max(len(r["options"]) for r in index)
        arrays: dict[str, np.ndarray] = {"option_mask": np.zeros((len(flat), k), bool)}
        for j, ((cap, q), meta) in enumerate(zip(flat, index)):
            n = len(meta["options"])
            arrays["option_mask"][j, :n] = True
            cap = {**cap, "probabilities": np.array(self._probabilities(cap["logits"], q), np.float32)}
            for key, value in cap.items():
                name = "option_logits" if key == "logits" else key
                if name not in arrays:
                    shape = (len(flat), k) + value.shape[1:] if key != "hidden_mean" else (len(flat),) + value.shape
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

    def lm(self, cache: bool = False):
        """DSPy's TypeSafe client for this backend. With cache=True, identical requests on the same weights reuse answers."""
        if self._client is None:
            from dspy_wsp.decision.client import DecisionTypeSafe
            from dspy_wsp.decision.server import SystemOneServer

            self._server = SystemOneServer(self)
            self._server.start()
            self._client = DecisionTypeSafe.for_backend(self)
            self._cached_client = self._client.copy(cache=True)
        return self._cached_client if cache else self._client

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

    family = "d1_omni"
    # The model's own code (*.py) is downloaded, never run, so a saved checkpoint loads with it.
    download_patterns = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "LICENSE", "*.py"]

    def __init__(self, model: str = "LiquidAI/d1-omni-600M", *, revision: str | None = "02b55d7076f15129e59ab3f94783f32c4b088674", framework: str | None = None, device: str | None = None, max_length: int | None = None):
        super().__init__(model, revision=revision, framework=framework, device=device, max_length=max_length)

    def _setup(self, max_length):
        from dspy_wsp.decision.families.d1_omni import encoding as d1

        self.config = d1.load_config(self._dir)
        self.tok = d1.Tokenizer(self._dir)
        self.max_length = int(max_length or self.config["max_length"])

    def _load_network(self):
        if self.framework == "torch":
            from dspy_wsp.decision.families.d1_omni.torch import D1Torch

            return D1Torch.from_dir(self._dir, device=self.device or "cpu")
        from dspy_wsp.decision.families.d1_omni.mlx import D1MLX

        return D1MLX.from_dir(self._dir)

    def _encode(self, state, q):
        from dspy_wsp.decision.families.d1_omni import encoding as d1

        try:
            ids, markers = d1.encode(self.tok, state, q, self.max_length)
        except ValueError as e:
            raise RequestError(str(e)) from None
        return ids, markers, q

    def _probabilities(self, logits, q):
        from dspy_wsp.decision.families.d1_omni import encoding as d1

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


class NanoJevBackend(DecisionBackend):
    """NanoJev (a Qwen3-0.6B backbone with a decision head). Defaults to C-Tianyu/NanoJev at a pinned revision."""

    family = "nanojev"
    download_patterns = ["best.safetensors", "config.json", "backbone_config/*", "tokenizer/*"]
    protected_description = "the tensors marked (out of scope) below (the token embedding and the backbone's final norm)"

    def __init__(self, model: str = "C-Tianyu/NanoJev", *, revision: str | None = "047b927b30882a1138fc504821b82ac145a4b81a", framework: str | None = None, device: str | None = None, max_length: int | None = 8192):
        super().__init__(model, revision=revision, framework=framework, device=device, max_length=max_length)

    def _setup(self, max_length):
        from dspy_wsp.decision.families.nanojev import encoding as nanojev

        self.run_config, self.backbone_config = nanojev.load_configs(self._dir)
        self.tok = nanojev.Tokenizer(self._dir / "tokenizer")
        self.max_length = int(max_length or self.run_config.get("max_length", 8192))

    def _load_network(self):
        if self.framework == "torch":
            from dspy_wsp.decision.families.nanojev.torch import NanoJevTorch

            return NanoJevTorch.from_dir(self._dir, device=self.device or "cpu", pad_token_id=self.tok.pad_token_id)
        from dspy_wsp.decision.families.nanojev.mlx import NanoJevMLX

        return NanoJevMLX.from_dir(self._dir, pad_token_id=self.tok.pad_token_id)

    def _encode(self, state, q):
        from dspy_wsp.decision.families.nanojev import encoding as nanojev

        paths, prefix = nanojev.encode(self.tok, state, q, self.max_length)
        return paths, prefix, q

    def _row_tokens(self, row) -> int:
        return sum(len(p) for p in row[0])

    def _probabilities(self, logits, q):
        from dspy_wsp.decision.families.nanojev import encoding as nanojev

        return nanojev.probabilities(logits, q)

    def default_protected(self):
        return (r"^backbone\.embed_tokens\.weight$", r"^backbone\.norm\.weight$")

    @property
    def tokenizer_dir(self) -> Path:
        return self._dir / "tokenizer"

    def num_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))

    def dims(self) -> dict:
        c = self.backbone_config
        return {"layers": c["num_hidden_layers"], "hidden": c["hidden_size"], "intermediate": c["intermediate_size"],
                "heads": c["num_attention_heads"], "kv_heads": c["num_key_value_heads"], "vocab": c["vocab_size"]}

    def describe(self) -> str:
        c = self.backbone_config
        n = sum(int(np.prod(self.reader.shape(k))) for k in self._param_for)
        return (
            f"{self.model_id} (NanoJev decision model): a Qwen3 backbone (`backbone.*`, {c['num_hidden_layers']} causal transformer "
            f"blocks, width {c['hidden_size']}, MLP width {c['intermediate_size']}, {c['num_attention_heads']} attention heads, "
            f"{c['num_key_value_heads']} key/value heads) and a decision head. Every option is read on its own text path: "
            "'State:\n<state>\nQuestion type: <choice, score or boolean>\nQuestion:\n<instructions>\nCandidate:\n<option text>\nDecision:' "
            "then EOS. The backbone's final hidden state at the EOS goes through a LayerNorm (`norm`) and a scalar head "
            "(`scalar`, one weight per hidden unit and a bias) to give the option's logit. For choice questions, a set-attention "
            "correction over the question's options (`set_project`, `set_attention`, `set_output`) adds to each logit. A noul "
            "(boolean) has one path, \"The proposition is true.\", and logits [0, z] for [false, true]; a score's options are its "
            f"level descriptions. A softmax per question gives the probabilities. {n:,} parameters in float32."
        )

    def read_conventions(self) -> dict:
        return {**super().read_conventions(),
                "head_inputs": "each option path's final hidden state after `norm` (the input to `scalar`)",
                "path_ends": "each option path's final hidden state before `norm`",
                "set_delta": "the set-attention correction added to each option's logit (choice questions; 0 otherwise)",
                "hidden_mean": "hidden state after each listed layer (0 = embeddings, l = after block l-1, the last index after the final norm), averaged over the state and question tokens"}


class D1LMBackend(DecisionBackend):
    """Liquid AI's d1 models built on LFM3 (text only): d1-2B by default, at a pinned revision.

    The answer is read from the language model's output at each question's answer position, so there is no
    separate head: edits land in the backbone.
    """

    family = "d1_lm"
    # The model's own code (*.py) is downloaded, never run, so a saved checkpoint loads with it.
    download_patterns = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                         "calibration.json", "processor_config.json", "LICENSE", "*.py"]
    protected_description = "the tensors marked (out of scope) below (the token embedding, which is also the output layer, and the final norm)"

    def __init__(self, model: str = "LiquidAI/d1-2B", *, revision: str | None = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1",
                 framework: str | None = None, device: str | None = None, max_length: int | None = None):
        super().__init__(model, revision=revision, framework=framework, device=device, max_length=max_length)

    def _setup(self, max_length):
        import json

        from dspy_wsp.decision.families.d1_lm import encoding

        self.config = json.loads((self._dir / "config.json").read_text())
        self.tok = encoding.Tokenizer(self._dir)
        self.calibration = encoding.load_calibration(self._dir)
        self.set_size = int(self.config.get("d1_set_size", encoding.SET_SIZE))
        self.scheme = int(self.config.get("d1_codes", 1))
        self.max_length = int(max_length or 16_384)

    def _load_network(self):
        if self.framework == "torch":
            from dspy_wsp.decision.families.d1_lm.torch import D1LMTorch

            return D1LMTorch.from_dir(self._dir, device=self.device or "cpu", pad_token_id=self.tok.pad_token_id)
        from dspy_wsp.decision.families.d1_lm.mlx import D1LMMLX

        return D1LMMLX.from_dir(self._dir, pad_token_id=self.tok.pad_token_id)

    def _check_question(self, raw):
        from dspy_wsp.decision.families.d1_lm import encoding

        encoding.check_supported(raw)

    def _encode_request(self, state, questions):
        from dspy_wsp.decision.families.d1_lm import encoding

        names = list(questions)
        rows = encoding.encode_request(self.tok, state, list(questions.values()), max_length=self.max_length,
                                       set_size=self.set_size, scheme=self.scheme)
        return [(row, names[i * self.set_size : (i + 1) * self.set_size]) for i, row in enumerate(rows)]

    def _row_outputs(self, output, n):
        return output

    def _row_tokens(self, row) -> int:
        return len(row.ids)

    def _probabilities(self, logits, q):
        from dspy_wsp.decision.families.d1_lm import encoding

        return encoding.probabilities(logits, self.calibration.temperature(q))

    def _model_tensor(self, name):
        return name.startswith("model.language_model.")

    def default_protected(self):
        return (r"^model\.language_model\.embed_tokens\.weight$", r"^model\.language_model\.embedding_norm\.weight$")

    def num_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))

    def dims(self) -> dict:
        t = self.config["text_config"]
        return {"layers": t["num_hidden_layers"], "hidden": t["hidden_size"], "intermediate": t["intermediate_size"],
                "heads": t["num_attention_heads"], "kv_heads": t["num_key_value_heads"], "vocab": t["vocab_size"]}

    def describe(self) -> str:
        t = self.config["text_config"]
        kinds = t["layer_types"]
        n = sum(int(np.prod(self.reader.shape(k))) for k in self._param_for)
        last = len(kinds) - 1
        return (
            f"{self.model_id} (d1 decision model on LFM3, text path): a language model with {len(kinds)} blocks, width "
            f"{t['hidden_size']}. Blocks {[i for i, k in enumerate(kinds) if k == 'full_attention']} use full attention, blocks "
            f"{[i for i, k in enumerate(kinds) if k == 'sliding_attention']} sliding-window attention over {t['sliding_window']} "
            "tokens, and the others a gated causal short convolution; each block has pre- and post-norms and a SwiGLU MLP "
            "(`feed_forward.w1`, `w3`, `w2`). A request is one ChatML user turn: the state, then up to "
            f"{self.set_size} questions, each ending at the assistant header and an empty <think></think> block (its answer "
            "position). In the full-attention blocks, state tokens read every token and each question reads the state and "
            "itself; the convolution and sliding-window blocks read only earlier tokens. Options are coded A, B, C "
            "(choice), 0, 1, 2 (score) or yes/no (noul). The answer is a softmax, at the calibration temperature (default "
            f"{self.calibration.default:g}, per question type where one is set), of the scores at the answer position for the option tokens: the final normed hidden state "
            "(after `embedding_norm`) times each option token's row of `embed_tokens` (the output layer is tied to the "
            "embedding); an option scores its best form. There is no separate head: the embedding and the final norm are "
            f"protected, so edits land in the blocks. The last write into the residual stream is "
            f"`layers.{last}.feed_forward.w2.weight`; its output passes through `layers.{last}.post_ffn_norm` (an RMS norm "
            "with a gain per dimension) before it joins the residual, and the sum passes through `embedding_norm` before the "
            f"readout scores it. {n:,} parameters in float32."
        )

    def read_conventions(self) -> dict:
        return {**super().read_conventions(),
                "option_logits": "each option's score at the answer position before the temperature: head_inputs . option_rows",
                "head_inputs": "the final normed hidden state at the question's answer position, one copy per option",
                "option_rows": "each option's best-form row of embed_tokens, which the readout multiplies by head_inputs",
                "hidden_mean": "hidden state after each listed layer (0 = embeddings, l = after block l-1), averaged over the state's tokens"}
