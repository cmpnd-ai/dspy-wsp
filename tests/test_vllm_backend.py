"""VLLMBackend against a stand-in vLLM engine, on the tiny random Llama (no GPU needed)."""

import asyncio
import sys
import threading
import time
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from dspy_wsp.backends._vllm_worker import checkpoint_with_edits  # noqa: E402
from dspy_wsp.tensors import TensorReader  # noqa: E402

L1 = "model.layers.1.mlp.down_proj.weight"


@dataclass
class _Out:
    token_ids: list
    finish_reason: str


@dataclass
class _RequestOut:
    outputs: list


class FakeEngine:
    """Answers with the prompt's last three tokens, then EOS; records weight reloads."""

    eos = None

    def __init__(self, args):
        self.args = args
        self.events: list = []
        self.loaded: dict = {}
        self.running = 0

    async def generate(self, prompt, params, request_id):
        self.running += 1
        self.events.append(("generate", params.seed))
        await asyncio.sleep(0.05)
        self.running -= 1
        ids = prompt["prompt_token_ids"][-3:]
        if params.max_tokens <= len(ids):
            yield _RequestOut([_Out(ids[: params.max_tokens], "length")])
        else:
            yield _RequestOut([_Out(ids + [self.eos], "stop")])

    async def collective_rpc(self, method, args=()):
        assert self.running == 0, "weights swapped while a request was running"
        self.events.append((method, args[1] is not None))
        self.loaded = dict(checkpoint_with_edits(*args))

    async def reset_prefix_cache(self):
        self.events.append(("reset_prefix_cache",))
        return True

    def shutdown(self):
        pass


@dataclass
class _Params:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    seed: int | None = None
    detokenize: bool = True
    extra: dict = field(default_factory=dict)


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.AsyncEngineArgs = lambda **kw: kw
    mod.AsyncLLMEngine = types.SimpleNamespace(from_engine_args=FakeEngine)
    mod.SamplingParams = _Params
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


@pytest.fixture
def backend(fake_vllm, tiny_model):
    from dspy_wsp import VLLMBackend

    b = VLLMBackend(str(tiny_model), max_tokens=12, concurrency=8, hf_kwargs={"device": "cpu"})
    FakeEngine.eos = b.tokenizer.eos_token_id
    yield b
    b.close()


def _edit(weights_dir):
    r = TensorReader(weights_dir)
    return {L1: r.get_f32(L1) + 0.5}


def test_checkpoint_with_edits_substitutes_only_edited_tensors(tiny_model, tmp_path):
    from safetensors.torch import save_file

    r = TensorReader(tiny_model)
    new = torch.full(r.shape(L1), 0.25, dtype=r.get(L1).dtype)
    path = tmp_path / "edits.safetensors"
    save_file({L1: new}, str(path))
    loaded = dict(checkpoint_with_edits(str(tiny_model), str(path)))
    assert sorted(loaded) == r.names
    assert torch.equal(loaded[L1], new)
    other = next(n for n in r.names if n != L1)
    assert torch.equal(loaded[other], r.get(other))
    save_file({"model.not_a_tensor": new}, str(path))
    with pytest.raises(KeyError):
        list(checkpoint_with_edits(str(tiny_model), str(path)))


def test_engine_args_name_the_worker_extension(backend):
    args = backend._engine.args
    assert args["worker_extension_cls"].endswith("WeightSwapExtension")
    assert args["max_model_len"] == 12 + 8192 and args["max_num_seqs"] == 8


def test_generate_strips_eos_and_reports_length(backend):
    prompt = backend.render_chat([{"role": "user", "content": "What is 2 + 2?"}], add_generation_prompt=True)
    call = backend.generate(prompt, seed=7)
    assert call.finish_reason == "stop" and call.output_tokens == 4
    assert call.completion == backend.tokenizer.decode(backend._encode(prompt)[-3:], skip_special_tokens=True)
    assert backend._engine.events[-1] == ("generate", 7)
    cut = backend.generate(prompt, max_tokens=2)
    assert cut.finish_reason == "length" and cut.output_tokens == 2


def test_swap_waits_for_requests_then_clears_prefix_cache(backend):
    prompt = "What is 2 + 2?"
    threads = [threading.Thread(target=backend.generate, args=(prompt,)) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.01)
    edit = _edit(backend.weights_dir)
    backend.set_weights(edit)
    for t in threads:
        t.join()
    kinds = [e[0] for e in backend._engine.events]
    i = kinds.index("wsp_load_weights")
    assert kinds[i + 1] == "reset_prefix_cache"
    loaded = backend._engine.loaded[L1].float().numpy()
    assert np.allclose(loaded, edit[L1], atol=1e-2)
    backend.set_weights(None)
    assert backend._engine.events[-2] == ("wsp_load_weights", False)
    assert torch.equal(backend._engine.loaded[L1], TensorReader(backend.weights_dir).get(L1))


def test_reads_go_to_hf_with_the_same_edits(backend):
    base, _ = backend.teacher_forced(["What is 2 + 2?"], token_logprobs=True)
    with backend.using(_edit(backend.weights_dir)):
        edited, _ = backend.teacher_forced(["What is 2 + 2?"], token_logprobs=True)
    again, _ = backend.teacher_forced(["What is 2 + 2?"], token_logprobs=True)
    assert not np.allclose(np.nan_to_num(base["token_logprobs"]), np.nan_to_num(edited["token_logprobs"]))
    assert np.array_equal(np.nan_to_num(base["token_logprobs"]), np.nan_to_num(again["token_logprobs"]))
    assert L1 in backend.editable_names()


def test_edits_made_before_hf_loads_reach_it(fake_vllm, tiny_model):
    from dspy_wsp import VLLMBackend

    b = VLLMBackend(str(tiny_model), max_tokens=12, hf_kwargs={"device": "cpu"})
    try:
        edit = _edit(b.weights_dir)
        b.set_weights(edit)
        assert b._hf is None
        p = b.hf._params[b.hf._param_for[L1]]
        assert np.allclose(p.detach().float().numpy(), edit[L1], atol=1e-2)
    finally:
        b.close()
