"""Apple-silicon backend on MLX, with continuous batching.

Every MLX operation runs on one worker thread: MLX streams belong to the thread
that made them. Program runs on other threads submit LM calls to that worker,
which serves them together through mlx-lm's `BatchGenerator`. Each sequence gets
its own seeded sampler, so an answer's randomness depends on its seed and not on
which other answers share its batch. (Batched matrix products can still differ
from unbatched ones in the last bits, so a seed fixes an answer exactly only for
the same batch composition.)

Edits arrive as float32 arrays in checkpoint layout. Some architectures change a
tensor when mlx-lm loads it (Qwen3.5 transposes its convolutions and adds 1 to its
RMSNorm weights). Those tensors take the edit as a difference from the original.
"""

from __future__ import annotations

import itertools
import json
import queue
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from dspy_wsp.backends.base import BackendEngine, LMCall, WeightBackend
from dspy_wsp.tensors import TensorReader

_ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*", "merges*", "vocab*"]
_WRAPPERS = {"model", "language_model"}


def _core(name: str) -> str:
    """A tensor name without the wrapper prefixes that differ between checkpoint and MLX ("model.", "language_model.")."""
    parts = name.split(".")
    while len(parts) > 1 and parts[0] in _WRAPPERS:
        parts = parts[1:]
    return ".".join(parts)


class _Tap:
    """Stands in for a module during a read: records its input, then calls it."""

    def __init__(self, inner, on_input):
        self._inner = inner
        self._on_input = on_input

    def __call__(self, x, *args, **kwargs):
        self._on_input(x)
        return self._inner(x, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


TIDY_EVERY = 256  # decode steps between cutting cache counter chains and clearing MLX's buffer cache


class _Sampler:
    """Temperature, top-k and top-p sampling for one sequence, from its own seed."""

    def __init__(self, seed: int | None, temperature: float, top_p: float, top_k: int):
        self.seed, self.temperature, self.top_p, self.top_k = seed, temperature, top_p, top_k
        self.step = 0

    def __call__(self, logprobs):
        import mlx.core as mx
        from mlx_lm.sample_utils import apply_top_k, apply_top_p

        if not self.temperature:
            return mx.argmax(logprobs, axis=-1)
        x = logprobs / self.temperature
        x = x - mx.logsumexp(x, axis=-1, keepdims=True)
        if self.top_k and self.top_k > 0:
            x = apply_top_k(x, self.top_k)
        if self.top_p and self.top_p < 1.0:
            x = apply_top_p(x, self.top_p)
        key = None
        if self.seed is not None:
            # A fresh key per step, not a split chain: a chain would grow MLX's lazy graph with every token.
            key = mx.random.key((self.seed * 1_000_003 + self.step) % (2**63 - 1))
        self.step += 1
        return mx.random.categorical(x, key=key)


class _PresencePenalty:
    """Subtracts `penalty` from every token already generated (vLLM's presence penalty; prompt tokens excluded)."""

    def __init__(self, penalty: float, vocab: int):
        self.penalty, self.vocab = penalty, vocab
        self.mask = None
        self.seen = 0

    def __call__(self, tokens, logits):
        import mlx.core as mx

        n = tokens.shape[0]
        if self.mask is None:
            self.mask = mx.zeros((logits.shape[-1],), dtype=logits.dtype)
            self.seen = n  # the first context is the whole prompt
        if n > self.seen:
            self.mask[tokens[self.seen : n]] = self.penalty
            self.seen = n
        return logits - self.mask


class MLXBackend(WeightBackend):
    """Run the student model with mlx-lm on Apple silicon, batching concurrent LM calls.

    Args:
        model: a Hugging Face model id or a local folder with safetensors weights.
        revision: hub revision to pin (recommended, so patches name an exact base).
        batch_size: how many sequences decode together. The forward-pass tool runs
            this many programs at once to fill the batch.
        max_tokens, temperature, top_p, top_k: default sampling for the program's LM calls.
        presence_penalty: subtracted from the logits of tokens already generated.
        chat_template_kwargs: passed to the chat template, e.g. {"enable_thinking": True}.
        split_thinking: return text before "</think>" as reasoning, not as the answer.
        prefill_batch_size: how many prompts are read together before decoding.
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        batch_size: int = 32,
        max_tokens: int = 32_768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        chat_template_kwargs: dict | None = None,
        split_thinking: bool = True,
        prefill_batch_size: int = 8,
    ):
        super().__init__()
        self._init_kwargs = {
            "model": model, "revision": revision, "batch_size": batch_size, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
            "chat_template_kwargs": chat_template_kwargs, "split_thinking": split_thinking,
            "prefill_batch_size": prefill_batch_size,
        }
        self.patch_dir: str | None = None  # set by WeightSpace.compile; saved with the program

        path = Path(model)
        if path.is_dir():
            self._dir = path.resolve()
        else:
            from huggingface_hub import snapshot_download

            self._dir = Path(snapshot_download(model, revision=revision, allow_patterns=_ALLOW))
        self._model_id = model
        self.revision = revision
        self.concurrency = int(batch_size)
        self.batch_size = int(batch_size)
        self.prefill_batch_size = int(prefill_batch_size)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.split_thinking = split_thinking
        self.reader = TensorReader(self._dir)
        config = json.loads((self._dir / "config.json").read_text())
        self.config = config
        self.text_config = config.get("text_config") or config
        self._lm = None

        self._tasks: queue.Queue = queue.Queue()
        self._uid_jobs: dict[int, dict] = {}  # sequences decoding now, by BatchGenerator uid
        self._dead: BaseException | None = None
        self._worker = threading.Thread(target=self._loop, name="wsp-mlx", daemon=True)
        ready: Future = Future()
        self._tasks.put(("init", ready))
        self._worker.start()
        ready.result()

    # --- the MLX thread -------------------------------------------------------------
    def _init_model(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.generate import BatchGenerator

        self.model, self.tokenizer = load(str(self._dir))
        self._params = dict(tree_flatten(self.model.parameters()))
        by_core: dict[str, list[str]] = {}
        for p in self._params:
            by_core.setdefault(_core(p), []).append(p)
        self._param_for = {}
        for name in self.reader.names:
            cands = by_core.get(_core(name), [])
            if len(cands) == 1 and int(np.prod(self.reader.shape(name))) == self._params[cands[0]].size:
                self._param_for[name] = cands[0]
        self._original = {p: self._params[p] for p in self._param_for.values()}
        self._layout: dict[str, tuple] = {}
        self._changed: set[str] = set()
        self._inner = self._find_inner()
        self._vocab = int(self.text_config.get("vocab_size") or self.tokenizer.vocab_size)
        self._eos = sorted(set(self.tokenizer.eos_token_ids))
        self._gen = BatchGenerator(
            self.model,
            max_tokens=self.max_tokens,
            stop_tokens=[[e] for e in self._eos],
            completion_batch_size=self.batch_size,
            prefill_batch_size=min(self.prefill_batch_size, self.batch_size),
        )
        # BatchGenerator wires GPU memory, which registers every buffer in a Metal residency set.
        # On long runs that registration can abort the process, and a small model gains little from it.
        mx.set_wired_limit(0)
        mx.eval(self.model.parameters())

    def _loop(self) -> None:
        try:
            self._serve()
        except BaseException as e:  # noqa: BLE001 - fail every waiting caller instead of hanging them
            self._dead = e
            for job in self._uid_jobs.values():
                job["future"].set_exception(e)
            while True:
                try:
                    item = self._tasks.get_nowait()
                except queue.Empty:
                    break
                (item[1]["future"] if item[0] == "gen" else item[1]).set_exception(e)
            raise

    def _submit(self, item) -> None:
        if self._dead is not None:
            raise RuntimeError("the MLX worker thread stopped") from self._dead
        self._tasks.put(item)

    def _serve(self) -> None:
        import mlx.core as mx

        pending_calls: list = []
        steps = 0
        while True:
            items = []
            try:
                items.append(self._tasks.get(block=not self._uid_jobs and not pending_calls))
                while True:
                    items.append(self._tasks.get_nowait())
            except queue.Empty:
                pass
            new_gens = [it[1] for it in items if it[0] == "gen"]
            pending_calls += [it for it in items if it[0] != "gen"]
            if new_gens:
                self._insert(new_gens)
            if pending_calls and not self._uid_jobs:
                # Calls (weight swaps, reads) wait until no sequence is decoding.
                for kind, fut, *rest in pending_calls:
                    try:
                        fut.set_result(self._init_model() if kind == "init" else rest[0]())
                    except BaseException as e:  # noqa: BLE001 - every failure goes back to the caller
                        fut.set_exception(e)
                pending_calls = []
                mx.clear_cache()
            elif self._uid_jobs:
                steps += 1
                if steps % TIDY_EVERY == 0:
                    self._cut_counter_chains()
                    mx.clear_cache()
                try:
                    self._step()
                except BaseException as e:  # noqa: BLE001
                    jobs, self._uid_jobs = self._uid_jobs, {}
                    self._gen.remove(list(jobs))
                    for job in jobs.values():
                        job["future"].set_exception(e)

    def _cut_counter_chains(self) -> None:
        """Evaluate the batch caches' per-step counters.

        mlx-lm's ArraysCache.advance subtracts from `left_padding` every step, but only one
        layer's copy is ever read. The others grow a lazy chain that holds one scalar buffer
        per step, and at about 29k steps the process hits Metal's buffer-count limit.
        """
        import mlx.core as mx

        counters = []
        for batch in (getattr(self._gen, "_prompt_batch", None), getattr(self._gen, "_generation_batch", None)):
            for c in getattr(batch, "prompt_cache", None) or []:
                for attr in ("left_padding", "lengths", "offset"):
                    v = getattr(c, attr, None)
                    if isinstance(v, mx.array):
                        counters.append(v)
        if counters:
            mx.eval(counters)

    def _insert(self, jobs: list[dict]) -> None:
        samplers = [_Sampler(j["seed"], j["temperature"], j["top_p"], j["top_k"]) for j in jobs]
        procs = [[_PresencePenalty(self.presence_penalty, self._vocab)] if self.presence_penalty else [] for _ in jobs]
        uids = self._gen.insert([j["ids"] for j in jobs], max_tokens=[j["max_tokens"] for j in jobs], samplers=samplers, logits_processors=procs)
        for uid, job in zip(uids, jobs):
            job["tokens"] = []
            self._uid_jobs[uid] = job

    def _step(self) -> None:
        for r in self._gen.next_generated():
            job = self._uid_jobs.get(r.uid)
            if job is None:
                continue
            job["tokens"].append(int(r.token))
            if r.finish_reason is not None:
                del self._uid_jobs[r.uid]
                job["future"].set_result((job["tokens"], r.finish_reason))

    def _call(self, fn):
        """Run `fn` on the MLX thread once no sequence is decoding, and return its result."""
        fut: Future = Future()
        self._submit(("call", fut, fn))
        return fut.result()

    # --- identity -------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def weights_dir(self) -> Path:
        return self._dir

    def dims(self) -> dict:
        c = self.text_config
        return {
            "layers": c.get("num_hidden_layers"),
            "hidden": c.get("hidden_size"),
            "intermediate": c.get("intermediate_size"),
            "heads": c.get("num_attention_heads"),
            "kv_heads": c.get("num_key_value_heads"),
            "vocab": c.get("vocab_size"),
            "tied": c.get("tie_word_embeddings", self.config.get("tie_word_embeddings")),
        }

    def attention_layers(self) -> list[int]:
        """Blocks with softmax attention (the only ones with attention probabilities to read)."""
        return [i for i, layer in enumerate(self._inner.layers) if hasattr(layer, "self_attn")]

    def describe(self) -> str:
        d = self.dims()
        n = sum(v.size for v in self._params.values())
        dtype = str(next(iter(self._params.values())).dtype).removeprefix("mlx.core.")
        attn = self.attention_layers()
        mixed = f" Blocks {attn} use softmax attention; the others use linear attention." if len(attn) < (d["layers"] or 0) else ""
        return (
            f"{self.model_id} ({self.text_config.get('model_type')}): {d['layers']} layers, width {d['hidden']}, MLP width {d['intermediate']}, "
            f"{d['heads']} attention heads" + (f" ({d['kv_heads']} key/value heads)" if d["kv_heads"] else "")
            + f", vocabulary {d['vocab']}, tied input/output embeddings: {d['tied']}; {n:,} parameters stored as {dtype}.{mixed}"
        )

    # --- weights --------------------------------------------------------------------
    def editable_names(self) -> list[str]:
        """Checkpoint tensors that map to a loaded parameter (others cannot be edited)."""
        return sorted(self._param_for)

    def _layout_for(self, name: str):
        """How a checkpoint tensor maps onto its MLX parameter: (axis order, whether the values are equal).

        Unequal values (e.g. RMSNorm weights stored with 1 added) take edits as a difference from the original.
        """
        if name not in self._layout:
            import mlx.core as mx

            w = self.reader.get_f32(name)
            p = np.array(self._original[self._param_for[name]].astype(mx.float32))
            for perm in itertools.permutations(range(w.ndim)):
                t = np.transpose(w, perm)
                if t.shape != p.shape:
                    continue
                if np.array_equal(t, p):
                    self._layout[name] = (perm, True)
                    break
                if np.allclose(t, p, rtol=1e-2, atol=1e-2) or np.allclose(t + 1.0, p, rtol=1e-2, atol=1e-2):
                    self._layout[name] = (perm, False)
                    break
            else:
                raise ValueError(f"cannot match {name} {w.shape} to MLX parameter {self._param_for[name]} {p.shape}")
        return self._layout[name]

    def _apply(self, edits: Mapping[str, np.ndarray]) -> None:
        for name in edits:
            if name not in self._param_for:
                raise KeyError(f"{name} does not map to a parameter of the loaded model")

        def apply():
            import mlx.core as mx

            updates = {p: self._original[p] for p in self._changed}
            for name, arr in edits.items():
                pname = self._param_for[name]
                orig = self._original[pname]
                perm, exact = self._layout_for(name)
                arr = np.asarray(arr, dtype=np.float32).reshape(self.reader.shape(name))
                if exact:
                    new = mx.array(np.transpose(arr, perm)).astype(orig.dtype)
                else:
                    delta = np.transpose(arr - self.reader.get_f32(name), perm)
                    new = (orig.astype(mx.float32) + mx.array(delta)).astype(orig.dtype)
                updates[pname] = new
            if updates:
                self.model.load_weights(list(updates.items()), strict=False)
                mx.eval(self.model.parameters())
            self._changed = {self._param_for[n] for n in edits}

        self._call(apply)

    def current_param(self, name: str) -> np.ndarray:
        """The loaded MLX parameter for a checkpoint tensor, as float32 in MLX layout."""

        def get():
            import mlx.core as mx
            from mlx.utils import tree_flatten

            return np.array(dict(tree_flatten(self.model.parameters()))[self._param_for[name]].astype(mx.float32))

        return self._call(get)

    # --- generation -----------------------------------------------------------------
    def lm(self):
        if self._lm is None:
            import dspy

            short = Path(self.model_id).name if Path(self.model_id).is_dir() else self.model_id
            self._lm = dspy.LM(f"wsp/{short}", engine=MLXEngine(self), cache=False, temperature=self.temperature, max_tokens=self.max_tokens)
        return self._lm

    def render_chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = False, **template_kwargs) -> str:
        kwargs = {**self.chat_template_kwargs, **template_kwargs}
        if not getattr(self.tokenizer, "chat_template", None):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return text + ("\nassistant:" if add_generation_prompt else "")
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer._tokenizer(text, add_special_tokens=False)["input_ids"]

    def num_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def generate(self, prompt: str, *, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None, stop=()) -> LMCall:
        ids = self._encode(prompt)
        max_tokens = max_tokens or self.max_tokens
        job = {
            "ids": ids,
            "max_tokens": int(max_tokens),
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "top_k": self.top_k if top_k is None else top_k,
            "seed": self._next_seed(seed),
            "future": Future(),
        }
        self._submit(("gen", job))
        tokens, finish = job["future"].result()
        new = tokens[:-1] if finish == "stop" and tokens and tokens[-1] in self._eos else tokens
        text = self.tokenizer._tokenizer.decode(new, skip_special_tokens=True)
        for st in stop or ():
            i = text.find(st)
            if i >= 0:
                text, finish = text[:i], "stop"
        call = LMCall(prompt=prompt, completion=text, finish_reason=finish, input_tokens=len(ids), output_tokens=len(tokens))
        self._record(call)
        return call

    # --- reading internals ----------------------------------------------------------
    def _find_inner(self):
        """The text model that holds the blocks, embeddings and final norm."""
        for _, m in self.model.named_modules():
            if isinstance(getattr(m, "layers", None), list) and "embed_tokens" in m and "norm" in m:
                return m
        raise ValueError("could not find the model's blocks (a module with layers, embed_tokens and norm)")

    def _head(self, h):
        for m in (self.model, getattr(self.model, "language_model", None)):
            if m is not None and "lm_head" in m:
                return m.lm_head(h)
        return self._inner.embed_tokens.as_linear(h)

    def _read(self, ids: list[int], *, hidden: Sequence[int], mlp: Sequence[int], attn: Sequence[int]):
        """One forward pass over `ids` (on the MLX thread). Returns (final normed states, captures)."""
        import mlx.core as mx

        inner = self._inner
        L = len(inner.layers)
        bad = sorted(set(attn) - set(self.attention_layers()))
        if bad:
            raise ValueError(f"blocks {bad} use linear attention and have no attention probabilities; softmax-attention blocks: {self.attention_layers()}")
        cap: dict = {}
        current = {"layer": None}
        saved_layers = list(inner.layers)
        restore = []
        attn_mod = None
        try:
            for i in range(L):
                def on_block(x, i=i):
                    current["layer"] = i
                    if i in hidden:
                        cap[("h", i)] = x
                inner.layers[i] = _Tap(saved_layers[i], on_block)
            if L in hidden:
                orig_norm = inner.norm
                inner["norm"] = _Tap(orig_norm, lambda x: None)
                restore.append(lambda: inner.__setitem__("norm", orig_norm))
            for i in mlp:
                block_mlp = saved_layers[i].mlp
                orig = block_mlp.down_proj
                block_mlp["down_proj"] = _Tap(orig, lambda x, i=i: cap.__setitem__(("m", i), x))
                restore.append(lambda block_mlp=block_mlp, orig=orig: block_mlp.__setitem__("down_proj", orig))
            if attn:
                import sys

                attn_mod = sys.modules[type(saved_layers[attn[0]].self_attn).__module__]
                orig_sdpa = attn_mod.scaled_dot_product_attention

                def sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
                    if current["layer"] not in attn:
                        return orig_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)
                    rep = q.shape[1] // k.shape[1]
                    if rep > 1:
                        k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
                    s = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
                    T, S = s.shape[-2], s.shape[-1]
                    if isinstance(mask, str) or (mask is None and T > 1):
                        causal = mx.arange(S)[None, :] <= (mx.arange(T)[:, None] + (S - T))
                        s = mx.where(causal, s, -mx.inf)
                    elif mask is not None:
                        s = mx.where(mask, s, -mx.inf) if mask.dtype == mx.bool_ else s + mask
                    p = mx.softmax(s, axis=-1)
                    cap[("a", current["layer"])] = p
                    return (p.astype(v.dtype) @ v)

                attn_mod.scaled_dot_product_attention = sdpa
                restore.append(lambda: setattr(attn_mod, "scaled_dot_product_attention", orig_sdpa))
            final = inner(mx.array([ids]))
            if L in hidden:
                cap[("h", L)] = final
            mx.eval(final, *cap.values())
        finally:
            for i in range(L):
                inner.layers[i] = saved_layers[i]
            for r in restore:
                r()
        return final, cap

    def _logprobs(self, final, ids: list[int], topk: int, chunk: int = 1024):
        import mlx.core as mx

        T = len(ids)
        lp = np.full(T, np.nan, np.float32)
        tk_ids = np.full((T, topk), -1, np.int32) if topk else None
        tk_lp = np.full((T, topk), np.nan, np.float32) if topk else None
        target = mx.array(ids)
        softcap = self.text_config.get("final_logit_softcapping")
        for s in range(0, T, chunk):
            logits = self._head(final[0, s : s + chunk]).astype(mx.float32)
            if softcap:
                logits = mx.tanh(logits / softcap) * softcap
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nxt = target[s + 1 : s + 1 + logp.shape[0]]
            if nxt.size:
                got = mx.take_along_axis(logp[: nxt.size], nxt[:, None], axis=1)[:, 0]
                lp[s + 1 : s + 1 + nxt.size] = np.array(got)
            if topk:
                idx = mx.argpartition(-logp, kth=topk - 1, axis=-1)[:, :topk]
                vals = mx.take_along_axis(logp, idx, axis=1)
                order = mx.argsort(-vals, axis=-1)
                tk_ids[s : s + logp.shape[0]] = np.array(mx.take_along_axis(idx, order, axis=1))
                tk_lp[s : s + logp.shape[0]] = np.array(mx.take_along_axis(vals, order, axis=1))
        return lp, tk_ids, tk_lp

    def teacher_forced(self, texts, *, token_logprobs=False, topk=0, hidden_states=(), mlp_act=(), attn_layers=()):
        enc = [self._encode(t) for t in texts]
        n, T = len(enc), max((len(e) for e in enc), default=0)
        d = self.dims()
        hidden_states, mlp_act, attn_layers = list(hidden_states), list(mlp_act), list(attn_layers)

        def run():
            import mlx.core as mx

            arrays: dict[str, np.ndarray] = {"token_ids": np.full((n, T), -1, np.int32), "lengths": np.array([len(e) for e in enc], np.int32)}
            if token_logprobs:
                arrays["token_logprobs"] = np.full((n, T), np.nan, np.float32)
            if topk:
                arrays["topk_ids"] = np.full((n, T, topk), -1, np.int32)
                arrays["topk_logprobs"] = np.full((n, T, topk), np.nan, np.float32)
            if hidden_states:
                arrays["hidden_states"] = np.zeros((n, len(hidden_states), T, d["hidden"]), np.float16)
            if mlp_act:
                arrays["mlp_act"] = np.zeros((n, len(mlp_act), T, d["intermediate"]), np.float16)
            if attn_layers:
                arrays["attentions"] = np.zeros((n, len(attn_layers), d["heads"], T, T), np.float16)
            for i, ids in enumerate(enc):
                if not ids:
                    continue
                L = len(ids)
                arrays["token_ids"][i, :L] = ids
                final, cap = self._read(ids, hidden=hidden_states, mlp=mlp_act, attn=attn_layers)
                if token_logprobs or topk:
                    lp, ti, tl = self._logprobs(final, ids, topk)
                    if token_logprobs:
                        arrays["token_logprobs"][i, :L] = lp
                    if topk:
                        arrays["topk_ids"][i, :L], arrays["topk_logprobs"][i, :L] = ti, tl
                for j, layer in enumerate(hidden_states):
                    arrays["hidden_states"][i, j, :L] = np.array(cap[("h", layer)][0].astype(mx.float32))
                for j, layer in enumerate(mlp_act):
                    arrays["mlp_act"][i, j, :L] = np.array(cap[("m", layer)][0].astype(mx.float32))
                for j, layer in enumerate(attn_layers):
                    arrays["attentions"][i, j, :, :L, :L] = np.array(cap[("a", layer)][0])
            return arrays

        arrays = self._call(run)
        meta = {
            "texts": n,
            "max_tokens": T,
            "hidden_state_layers": hidden_states,
            "mlp_layers": mlp_act,
            "attention_layers": attn_layers,
            "mlp_module": "mlp.down_proj",
            "conventions": {
                "padding": "right-padded to the longest text; token_ids = -1 marks padding",
                "token_logprobs[t]": "log-prob of token t given tokens < t (t = 0 is NaN)",
                "topk[t]": "the k most likely next tokens after reading tokens <= t",
                "hidden_states": "index l = residual stream after block l-1 (0 = embeddings); the last index is after the final norm",
                "mlp_act": "input to each listed block's MLP output projection (block indices as in tensor names)",
                "attentions": f"[text, layer, head, query, key] probabilities; only softmax-attention blocks {self.attention_layers()} have them",
            },
        }
        return arrays, meta

    def pooled_activations(self, pairs, *, layers, mlp_layers=(), max_answer_tokens=8192):
        layers, mlp_layers = list(layers or []), list(mlp_layers or [])
        d = self.dims()
        N = len(pairs)
        enc = [(self._encode(p), self._encode(c)[:max_answer_tokens]) for p, c in pairs]

        def run():
            import mlx.core as mx

            out = {}
            if layers:
                out["hidden_last_prompt"] = np.zeros((N, len(layers), d["hidden"]), np.float32)
                out["hidden_mean_answer"] = np.full((N, len(layers), d["hidden"]), np.nan, np.float32)
            if mlp_layers:
                out["mlp_last_prompt"] = np.zeros((N, len(mlp_layers), d["intermediate"]), np.float32)
                out["mlp_mean_answer"] = np.full((N, len(mlp_layers), d["intermediate"]), np.nan, np.float32)
            for i, (p, c) in enumerate(enc):
                if not p:
                    continue
                _, cap = self._read(p + c, hidden=layers, mlp=mlp_layers, attn=[])
                last, ans = len(p) - 1, slice(len(p), len(p) + len(c))
                for key, idx, name in [("h", layers, "hidden"), ("m", mlp_layers, "mlp")]:
                    for j, layer in enumerate(idx):
                        a = cap[(key, layer)][0].astype(mx.float32)
                        out[f"{name}_last_prompt"][i, j] = np.array(a[last])
                        if c:
                            out[f"{name}_mean_answer"][i, j] = np.array(a[ans].mean(axis=0))
            return out

        return self._call(run)


class MLXEngine(BackendEngine):
    backend_class = MLXBackend
