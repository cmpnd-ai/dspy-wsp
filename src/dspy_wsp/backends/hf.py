"""In-process Hugging Face transformers backend.

Simple and exact, but it generates one sequence at a time. It suits small
models and development; long-thinking runs at the paper's scale (512 answers of
up to 32k tokens per pass) want a batched server backend such as vLLM.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from dspy_wsp.backends.base import LOADED_BACKENDS, BackendEngine, LMCall, WeightBackend
from dspy_wsp.tensors import TensorReader

_ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*", "merges*", "vocab*"]
_BLOCK = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.(.+)$")


class HFBackend(WeightBackend):
    """Run the student model with transformers, in this process.

    Args:
        model: a Hugging Face model id or a local folder with safetensors weights.
        revision: hub revision to pin (recommended, so patches name an exact base).
        device: "cuda", "mps" or "cpu" (default: cuda if available).
        dtype: "auto" keeps the checkpoint's dtype.
        max_tokens, temperature, top_p, top_k: default sampling for the program's LM calls.
        chat_template_kwargs: passed to the chat template, e.g. {"enable_thinking": True}.
        split_thinking: return text before "</think>" as reasoning, not as the answer.
        generation_kwargs: extra arguments for `model.generate`.
        attn_implementation: "eager" is required to read attention probabilities.
        mlp_module: the module whose input is "the MLP activation", relative to a block.
            Detected automatically for Llama/Qwen ("mlp.down_proj") and GPT-2 ("mlp.c_proj").
    """

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | torch.dtype = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        chat_template_kwargs: dict | None = None,
        split_thinking: bool = True,
        generation_kwargs: dict | None = None,
        attn_implementation: str | None = None,
        mlp_module: str | None = None,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._init_kwargs = {
            "model": model, "revision": revision, "device": device, "dtype": dtype if isinstance(dtype, str) else str(dtype).removeprefix("torch."),
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "chat_template_kwargs": chat_template_kwargs, "split_thinking": split_thinking, "generation_kwargs": generation_kwargs,
            "attn_implementation": attn_implementation, "mlp_module": mlp_module, "trust_remote_code": trust_remote_code,
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.split_thinking = split_thinking
        self.generation_kwargs = dict(generation_kwargs or {})

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else dtype
        load_kwargs = {"trust_remote_code": trust_remote_code}
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        load_kwargs[dtype_key] = torch_dtype
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(str(self._dir), **load_kwargs).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self._dir), trust_remote_code=trust_remote_code)
        self.reader = TensorReader(self._dir)

        self._params = dict(self.model.named_parameters())
        self._param_for = self._map_params()
        self._saved: dict[str, torch.Tensor] = {}
        self._mlp_module = mlp_module or self._detect_mlp_module()
        self._lm = None
        eos = self.model.generation_config.eos_token_id
        self._eos = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
        if self.tokenizer.eos_token_id is not None:
            self._eos.add(self.tokenizer.eos_token_id)

    # --- identity -------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def weights_dir(self) -> Path:
        return self._dir

    @property
    def text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", None) or cfg

    def dims(self) -> dict:
        c = self.text_config
        return {
            "layers": getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", None),
            "hidden": getattr(c, "hidden_size", None) or getattr(c, "n_embd", None),
            "intermediate": getattr(c, "intermediate_size", None) or getattr(c, "n_inner", None) or 4 * (getattr(c, "n_embd", 0) or 0),
            "heads": getattr(c, "num_attention_heads", None) or getattr(c, "n_head", None),
            "kv_heads": getattr(c, "num_key_value_heads", None),
            "vocab": getattr(c, "vocab_size", None),
            "tied": getattr(self.model.config, "tie_word_embeddings", None),
        }

    def describe(self) -> str:
        d = self.dims()
        n = sum(p.numel() for p in self.model.parameters())
        dtype = next(self.model.parameters()).dtype
        return (
            f"{self.model_id} ({self.text_config.model_type}): {d['layers']} layers, width {d['hidden']}, MLP width {d['intermediate']}, "
            f"{d['heads']} attention heads" + (f" ({d['kv_heads']} key/value heads)" if d["kv_heads"] else "")
            + f", vocabulary {d['vocab']}, tied input/output embeddings: {d['tied']}; {n:,} parameters stored as {str(dtype).removeprefix('torch.')}."
        )

    # --- weights --------------------------------------------------------------------
    def _map_params(self) -> dict[str, str]:
        mapping = {}
        for name in self.reader.names:
            if name in self._params:
                mapping[name] = name
                continue
            # Vision-language checkpoints (e.g. Qwen3.5) nest the text model under "language_model."
            text_name = name.replace("language_model.", "", 1)
            if text_name != name and text_name in self._params:
                mapping[name] = text_name
                continue
            cands = [p for p in self._params if p.endswith("." + name) or name.endswith("." + p)]
            if len(cands) == 1:
                mapping[name] = cands[0]
        return mapping

    def editable_names(self) -> list[str]:
        """Checkpoint tensors that map to a loaded parameter (others cannot be edited)."""
        return sorted(self._param_for)

    def _apply(self, edits: Mapping[str, np.ndarray]) -> None:
        with torch.no_grad():
            for pname, original in self._saved.items():
                self._params[pname].data.copy_(original)
            self._saved = {}
            for name, arr in edits.items():
                if name not in self._param_for:
                    raise KeyError(f"{name} does not map to a parameter of the loaded model")
                p = self._params[self._param_for[name]]
                self._saved[self._param_for[name]] = p.data.detach().clone()
                p.data.copy_(torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(p.shape).to(p.dtype))

    # --- generation -----------------------------------------------------------------
    def lm(self):
        if self._lm is None:
            import dspy

            short = Path(self.model_id).name if Path(self.model_id).is_dir() else self.model_id
            self._lm = dspy.LM(f"wsp/{short}", engine=HFEngine(self), cache=False, temperature=self.temperature, max_tokens=self.max_tokens)
        return self._lm

    def render_chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = False, **template_kwargs) -> str:
        kwargs = {**self.chat_template_kwargs, **template_kwargs}
        if not getattr(self.tokenizer, "chat_template", None):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return text + ("\nassistant:" if add_generation_prompt else "")
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)

    def num_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt: str, *, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None, stop=()) -> LMCall:
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else next(iter(self._eos), 0)
        kw = dict(max_new_tokens=max_tokens, pad_token_id=pad, **self.generation_kwargs)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=self.top_p if top_p is None else top_p, top_k=self.top_k if top_k is None else top_k)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        s = self._next_seed(seed)
        if s is not None:
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        new = out[0, enc["input_ids"].shape[1] :].tolist()
        while new and new[-1] == pad and pad not in self._eos:
            new.pop()
        ended = bool(new) and new[-1] in self._eos
        finish = "length" if len(new) >= max_tokens and not ended else "stop"
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        for st in stop or ():
            i = text.find(st)
            if i >= 0:
                text, finish = text[:i], "stop"
        call = LMCall(prompt=prompt, completion=text, finish_reason=finish, input_tokens=int(enc["input_ids"].shape[1]), output_tokens=len(new))
        self._record(call)
        return call

    # --- reading internals ----------------------------------------------------------
    def _detect_mlp_module(self) -> str | None:
        names = [n for n, _ in self.model.named_modules()]
        for cand in ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h"):
            if any(n.endswith(cand) for n in names):
                return cand
        return None

    def _mlp_modules(self, layers: Sequence[int]) -> dict[int, torch.nn.Module]:
        if not layers:
            return {}
        if self._mlp_module is None:
            raise ValueError("could not find the MLP output projection; pass mlp_module= to HFBackend")
        found = {}
        for name, mod in self.model.named_modules():
            m = _BLOCK.search(name)
            if m and m.group(2) == self._mlp_module and int(m.group(1)) in layers:
                found.setdefault(int(m.group(1)), mod)
        missing = sorted(set(layers) - set(found))
        if missing:
            raise ValueError(f"no MLP module for layers {missing}")
        return found

    def _run(self, ids: list[int], *, hidden: bool, mlp_layers: Sequence[int], attn: bool):
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for layer, mod in self._mlp_modules(mlp_layers).items():
            hooks.append(mod.register_forward_pre_hook(lambda m, inp, layer=layer: captured.__setitem__(layer, inp[0].detach())))
        try:
            x = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                out = self.model.base_model(input_ids=x, output_hidden_states=hidden, output_attentions=attn, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return out, captured

    @torch.no_grad()
    def _logprobs(self, last_hidden: torch.Tensor, ids: list[int], topk: int, chunk: int = 512):
        """Log-probs of each next token, computed in chunks so the vocabulary never fills memory at once."""
        head = self.model.get_output_embeddings()
        softcap = getattr(self.text_config, "final_logit_softcapping", None)
        T = len(ids)
        lp = np.full(T, np.nan, np.float32)
        tk_ids = np.full((T, topk), -1, np.int32) if topk else None
        tk_lp = np.full((T, topk), np.nan, np.float32) if topk else None
        target = torch.tensor(ids, device=last_hidden.device)
        for s in range(0, T, chunk):
            h = last_hidden[0, s : s + chunk]
            logits = head(h).float()
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            logp = torch.log_softmax(logits, dim=-1)
            # position t predicts token t+1
            nxt = target[s + 1 : s + 1 + logp.shape[0]]
            if nxt.numel():
                lp[s + 1 : s + 1 + nxt.numel()] = logp[: nxt.numel()].gather(1, nxt[:, None])[:, 0].cpu().numpy()
            if topk:
                v, i = logp.topk(topk, dim=-1)
                tk_ids[s : s + logp.shape[0]] = i.cpu().numpy()
                tk_lp[s : s + logp.shape[0]] = v.cpu().numpy()
        return lp, tk_ids, tk_lp

    @torch.no_grad()
    def teacher_forced(self, texts, *, token_logprobs=False, topk=0, hidden_states=(), mlp_act=(), attn_layers=()):
        enc = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
        n, T = len(enc), max((len(e) for e in enc), default=0)
        d = self.dims()
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
            self._ensure_eager()
        for i, ids in enumerate(enc):
            if not ids:
                continue
            L = len(ids)
            arrays["token_ids"][i, :L] = ids
            out, mlp = self._run(ids, hidden=bool(hidden_states), mlp_layers=mlp_act, attn=bool(attn_layers))
            if token_logprobs or topk:
                lp, ti, tl = self._logprobs(out.last_hidden_state, ids, topk)
                if token_logprobs:
                    arrays["token_logprobs"][i, :L] = lp
                if topk:
                    arrays["topk_ids"][i, :L], arrays["topk_logprobs"][i, :L] = ti, tl
            for j, layer in enumerate(hidden_states):
                arrays["hidden_states"][i, j, :L] = out.hidden_states[layer][0].float().cpu().numpy()
            for j, layer in enumerate(mlp_act):
                arrays["mlp_act"][i, j, :L] = mlp[layer][0].float().cpu().numpy()
            for j, layer in enumerate(attn_layers):
                arrays["attentions"][i, j, :, :L, :L] = out.attentions[layer][0].float().cpu().numpy()
        meta = {
            "texts": n,
            "max_tokens": T,
            "hidden_state_layers": list(hidden_states),
            "mlp_layers": list(mlp_act),
            "attention_layers": list(attn_layers),
            "mlp_module": self._mlp_module,
            "conventions": {
                "padding": "right-padded to the longest text; token_ids = -1 marks padding",
                "token_logprobs[t]": "log-prob of token t given tokens < t (t = 0 is NaN)",
                "topk[t]": "the k most likely next tokens after reading tokens <= t",
                "hidden_states": "index l = residual stream after block l-1 (0 = embeddings); the last index is after the final norm",
                "mlp_act": "input to each listed block's MLP output projection (block indices as in tensor names)",
                "attentions": "[text, layer, head, query, key] probabilities",
            },
        }
        return arrays, meta

    def _ensure_eager(self) -> None:
        impl = getattr(self.model.config, "_attn_implementation", None)
        if impl == "eager":
            return
        setter = getattr(self.model, "set_attn_implementation", None)
        if setter is None:
            raise ValueError("reading attention needs HFBackend(..., attn_implementation='eager')")
        setter("eager")

    @torch.no_grad()
    def pooled_activations(self, pairs, *, layers, mlp_layers=(), max_answer_tokens=8192):
        mlp_layers = list(mlp_layers or [])
        d = self.dims()
        N = len(pairs)
        out = {}
        if layers:
            out["hidden_last_prompt"] = np.zeros((N, len(layers), d["hidden"]), np.float32)
            out["hidden_mean_answer"] = np.full((N, len(layers), d["hidden"]), np.nan, np.float32)
        if mlp_layers:
            out["mlp_last_prompt"] = np.zeros((N, len(mlp_layers), d["intermediate"]), np.float32)
            out["mlp_mean_answer"] = np.full((N, len(mlp_layers), d["intermediate"]), np.nan, np.float32)
        for i, (prompt, completion) in enumerate(pairs):
            p = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c = self.tokenizer(completion, add_special_tokens=False)["input_ids"][:max_answer_tokens]
            if not p:
                continue
            res, cap = self._run(p + c, hidden=bool(layers), mlp_layers=mlp_layers, attn=False)
            last, ans = len(p) - 1, slice(len(p), len(p) + len(c))
            for j, layer in enumerate(layers):
                h = res.hidden_states[layer][0].float()
                out["hidden_last_prompt"][i, j] = h[last].cpu().numpy()
                if c:
                    out["hidden_mean_answer"][i, j] = h[ans].mean(0).cpu().numpy()
            for j, layer in enumerate(mlp_layers):
                a = cap[layer][0].float()
                out["mlp_last_prompt"][i, j] = a[last].cpu().numpy()
                if c:
                    out["mlp_mean_answer"][i, j] = a[ans].mean(0).cpu().numpy()
        return out


class HFEngine(BackendEngine):
    backend_class = HFBackend


_LOADED = LOADED_BACKENDS
