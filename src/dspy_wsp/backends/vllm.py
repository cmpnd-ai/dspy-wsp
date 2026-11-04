"""NVIDIA backend: vLLM generates, Hugging Face transformers reads internals.

vLLM's engine batches every concurrent LM call, which makes long-thinking runs
at the paper's scale practical: 512 answers of up to 32k tokens per pass. It
cannot return hidden states or MLP activations, so teacher-forced reads go to an
`HFBackend` on the same GPU. Both models hold the same edits.

Edits reach vLLM through a worker extension that reloads the whole checkpoint
in place, with edited tensors substituted. Reloading everything costs a second
or two for a small model and is safe for fused layers, which vLLM rebuilds from
scratch on any reload. Swaps wait until no request is running, then clear the
prefix cache, because cached prompt states belong to the old weights.

Each request samples from its own seed. Batched kernels can still change the
last bits of the logits with the batch's makeup; pass `batch_invariant=True`
to make an answer depend only on its prompt and seed, at some cost in speed.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from dspy_wsp.backends.base import BackendEngine, LMCall, WeightBackend
from dspy_wsp.tensors import TensorReader

_ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*", "merges*", "vocab*"]
_EXTENSION = "dspy_wsp.backends._vllm_worker.WeightSwapExtension"


class VLLMBackend(WeightBackend):
    """Run the student model with vLLM on NVIDIA GPUs, batching concurrent LM calls.

    Args:
        model: a Hugging Face model id or a local folder with safetensors weights.
        revision: hub revision to pin (recommended, so patches name an exact base).
        concurrency: how many program runs the forward-pass tool keeps in flight.
            vLLM schedules them itself, so this can match a whole pass (the paper's 512).
        max_tokens, temperature, top_p, top_k: default sampling for the program's LM calls.
        presence_penalty: subtracted from the logits of tokens already generated.
        chat_template_kwargs: passed to the chat template, e.g. {"enable_thinking": True}.
        split_thinking: return text before "</think>" as reasoning, not as the answer.
        gpu_memory_utilization: vLLM's share of GPU memory. The rest holds the
            Hugging Face model for teacher-forced reads.
        max_model_len: longest prompt plus answer (default: max_tokens + 8192).
        batch_invariant: make each answer independent of what else is in the batch.
        engine_kwargs: extra arguments for vLLM's `AsyncEngineArgs`.
        hf_kwargs: extra arguments for the `HFBackend` that serves teacher-forced reads.
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        concurrency: int = 512,
        max_tokens: int = 32_768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        chat_template_kwargs: dict | None = None,
        split_thinking: bool = True,
        gpu_memory_utilization: float = 0.6,
        max_model_len: int | None = None,
        batch_invariant: bool = False,
        engine_kwargs: dict | None = None,
        hf_kwargs: dict | None = None,
    ):
        super().__init__()
        self._init_kwargs = {
            "model": model, "revision": revision, "concurrency": concurrency, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
            "chat_template_kwargs": chat_template_kwargs, "split_thinking": split_thinking,
            "gpu_memory_utilization": gpu_memory_utilization, "max_model_len": max_model_len,
            "batch_invariant": batch_invariant, "engine_kwargs": engine_kwargs, "hf_kwargs": hf_kwargs,
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
        self.concurrency = int(concurrency)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.split_thinking = split_thinking
        self.reader = TensorReader(self._dir)
        self.config = json.loads((self._dir / "config.json").read_text())
        self.text_config = self.config.get("text_config") or self.config
        self._hf_kwargs = dict(hf_kwargs or {})
        self._hf = None
        self._lm = None

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self._dir))
        self._eos = self._eos_ids()

        # Requests run on one asyncio loop in a background thread; program threads submit to it.
        self._gate = threading.Condition()
        self._running = 0
        self._swapping = False
        self._swap_lock = threading.Lock()
        self._tmp = Path(tempfile.mkdtemp(prefix="wsp-vllm-"))
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="wsp-vllm", daemon=True)
        self._thread.start()

        if batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        args = {
            "model": str(self._dir),
            "tokenizer": str(self._dir),
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len or max_tokens + 8192,
            "max_num_seqs": max(1, min(self.concurrency, 1024)),
            "worker_extension_cls": _EXTENSION,
            **(engine_kwargs or {}),
        }
        self._engine = self._run(self._start_engine(args))

    async def _start_engine(self, args: dict):
        # Built inside the loop so the engine's output handler binds to it.
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**args))

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _eos_ids(self) -> set[int]:
        eos: set[int] = set()
        gen_file = self._dir / "generation_config.json"
        if gen_file.exists():
            e = json.loads(gen_file.read_text()).get("eos_token_id")
            eos.update(e if isinstance(e, list) else [e] if e is not None else [])
        if self.tokenizer.eos_token_id is not None:
            eos.add(self.tokenizer.eos_token_id)
        return eos

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

    def describe(self) -> str:
        return self.hf.describe()

    # --- weights --------------------------------------------------------------------
    @property
    def hf(self):
        """The Hugging Face copy of the model that serves teacher-forced reads, loaded on first use."""
        if self._hf is None:
            from dspy_wsp.backends.hf import HFBackend

            kw = {"device": "cuda", "max_tokens": self.max_tokens, "chat_template_kwargs": self.chat_template_kwargs, **self._hf_kwargs}
            self._hf = HFBackend(str(self._dir), **kw)
            if self._edits:
                self._hf.set_weights(self._edits)
        return self._hf

    def editable_names(self) -> list[str]:
        return self.hf.editable_names()

    def _write_edits(self, edits: Mapping[str, np.ndarray]) -> str | None:
        """Save edits as checkpoint-dtype tensors for the vLLM workers to read."""
        if not edits:
            return None
        import torch
        from safetensors.torch import save_file

        tensors = {}
        for name, arr in edits.items():
            if name not in self.reader:
                raise KeyError(f"{name} is not a tensor of {self.model_id}")
            dtype = self.reader.get(name).dtype
            tensors[name] = torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(self.reader.shape(name)).to(dtype).contiguous()
        path = self._tmp / f"edits-{uuid.uuid4().hex}.safetensors"
        save_file(tensors, str(path))
        return str(path)

    def _apply(self, edits: Mapping[str, np.ndarray]) -> None:
        with self._swap_lock:
            path = self._write_edits(edits)
            with self._gate:
                self._swapping = True
                self._gate.wait_for(lambda: self._running == 0)
            try:
                self._run(self._swap(path))
                if self._hf is not None:
                    self._hf.set_weights(edits)
            finally:
                with self._gate:
                    self._swapping = False
                    self._gate.notify_all()
                if path:
                    Path(path).unlink(missing_ok=True)

    async def _swap(self, edits_path: str | None) -> None:
        await self._engine.collective_rpc("wsp_load_weights", args=(str(self._dir), edits_path))
        await self._engine.reset_prefix_cache()

    # --- generation -----------------------------------------------------------------
    def lm(self):
        if self._lm is None:
            import dspy

            short = Path(self.model_id).name if Path(self.model_id).is_dir() else self.model_id
            self._lm = dspy.LM(f"wsp/{short}", engine=VLLMEngine(self), cache=False, temperature=self.temperature, max_tokens=self.max_tokens)
        return self._lm

    def render_chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = False, **template_kwargs) -> str:
        kwargs = {**self.chat_template_kwargs, **template_kwargs}
        if not getattr(self.tokenizer, "chat_template", None):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return text + ("\nassistant:" if add_generation_prompt else "")
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def num_tokens(self, text: str) -> int:
        return len(self._encode(text))

    async def _generate(self, ids: list[int], params) -> tuple[list[int], str]:
        final = None
        async for out in self._engine.generate({"prompt_token_ids": ids}, params, request_id=uuid.uuid4().hex):
            final = out
        o = final.outputs[0]
        return list(o.token_ids), o.finish_reason or "stop"

    def generate(self, prompt: str, *, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None, stop=()) -> LMCall:
        from vllm import SamplingParams

        ids = self._encode(prompt)
        max_tokens = int(max_tokens or self.max_tokens)
        temperature = self.temperature if temperature is None else temperature
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature or 0.0,
            top_p=self.top_p if top_p is None else top_p,
            top_k=self.top_k if top_k is None else top_k,
            presence_penalty=self.presence_penalty,
            seed=self._next_seed(seed),
            detokenize=False,
        )
        with self._gate:
            self._gate.wait_for(lambda: not self._swapping)
            self._running += 1
        try:
            tokens, finish = self._run(self._generate(ids, params))
        finally:
            with self._gate:
                self._running -= 1
                self._gate.notify_all()
        finish = "length" if finish == "length" else "stop"
        new = tokens[:-1] if finish == "stop" and tokens and tokens[-1] in self._eos else tokens
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        for st in stop or ():
            i = text.find(st)
            if i >= 0:
                text, finish = text[:i], "stop"
        call = LMCall(prompt=prompt, completion=text, finish_reason=finish, input_tokens=len(ids), output_tokens=len(tokens))
        self._record(call)
        return call

    # --- reading internals ----------------------------------------------------------
    def teacher_forced(self, texts, **kwargs):
        return self.hf.teacher_forced(texts, **kwargs)

    def pooled_activations(self, pairs, **kwargs):
        return self.hf.pooled_activations(pairs, **kwargs)

    def close(self) -> None:
        engine, self._engine = getattr(self, "_engine", None), None
        if engine is not None:
            engine.shutdown()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


class VLLMEngine(BackendEngine):
    backend_class = VLLMBackend
