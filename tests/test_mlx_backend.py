"""The MLX backend on the tiny random Llama, checked against the Hugging Face backend."""

import json
import threading

import dspy
import numpy as np
import pytest

pytest.importorskip("mlx_lm")
pytest.importorskip("transformers")

from conftest import scripted_writer  # noqa: E402

from dspy_wsp import HFBackend, MLXBackend, Patch, WeightSpace  # noqa: E402
from dspy_wsp.tensors import TensorReader  # noqa: E402

L1 = "model.layers.1.mlp.down_proj.weight"
TEXTS = ["What is 2 + 2?", "Respond with answer"]


@pytest.fixture(scope="module")
def mlx(tiny_model):
    return MLXBackend(str(tiny_model), max_tokens=12, batch_size=4)


@pytest.fixture(scope="module")
def hf(tiny_model):
    return HFBackend(str(tiny_model), device="cpu", max_tokens=12, attn_implementation="eager")


def _big_edit(backend):
    r = TensorReader(backend.weights_dir)
    g = np.random.default_rng(0)
    return {L1: r.get_f32(L1) + g.normal(0, 1.0, size=r.shape(L1)).astype(np.float32)}


def test_describe_and_names(mlx):
    assert "2 layers" in mlx.describe() and "width 32" in mlx.describe()
    assert L1 in mlx.editable_names()
    assert mlx.dims()["intermediate"] == 64
    assert mlx.attention_layers() == [0, 1]
    assert mlx.prefill_batch_size == 8  # softmax attention only, so prompts may be read together


def test_teacher_forced_matches_hf(mlx, hf):
    kw = dict(token_logprobs=True, topk=3, hidden_states=[0, 1, 2], mlp_act=[1], attn_layers=[0])
    a, _ = mlx.teacher_forced(TEXTS, **kw)
    b, _ = hf.teacher_forced(TEXTS, **kw)
    assert np.array_equal(a["token_ids"], b["token_ids"])
    for key in ("token_logprobs", "hidden_states", "mlp_act", "attentions"):
        assert a[key].shape == b[key].shape, key
        assert np.allclose(np.nan_to_num(a[key].astype(np.float32)), np.nan_to_num(b[key].astype(np.float32)), atol=5e-2), key
    assert (a["topk_ids"][:, :, 0] == b["topk_ids"][:, :, 0]).mean() > 0.9


def test_edits_apply_and_restore(mlx, hf):
    base, _ = mlx.teacher_forced(TEXTS[:1], token_logprobs=True)
    edit = _big_edit(mlx)
    with mlx.using(edit):
        edited, _ = mlx.teacher_forced(TEXTS[:1], token_logprobs=True)
        assert np.allclose(mlx.current_param(L1), edit[L1].astype(np.float32), atol=2e-2)
    with hf.using(edit):
        hf_edited, _ = hf.teacher_forced(TEXTS[:1], token_logprobs=True)
    assert not np.allclose(np.nan_to_num(edited["token_logprobs"]), np.nan_to_num(base["token_logprobs"]), atol=1e-3)
    assert np.allclose(np.nan_to_num(edited["token_logprobs"]), np.nan_to_num(hf_edited["token_logprobs"]), atol=5e-2)
    again, _ = mlx.teacher_forced(TEXTS[:1], token_logprobs=True)
    assert np.array_equal(np.nan_to_num(again["token_logprobs"]), np.nan_to_num(base["token_logprobs"]))


def test_seeded_generation(mlx):
    prompt = mlx.render_chat([{"role": "user", "content": "hello"}], add_generation_prompt=True)
    a = mlx.generate(prompt, seed=7).completion
    b = mlx.generate(prompt, seed=7).completion
    assert a == b
    texts = {mlx.generate(prompt, seed=s).completion for s in range(6)}
    assert len(texts) > 1  # different seeds sample differently
    with mlx.record_calls() as calls:
        mlx.generate(prompt, seed=1)
    assert calls[0].output_tokens <= 12 and calls[0].prompt.endswith("<s>assistant\n")


def test_concurrent_calls_share_batches_and_record_per_thread(mlx):
    prompt = mlx.render_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    results, recorded = {}, {}

    def run(i):
        mlx.seed(100 + i)
        with mlx.record_calls() as calls:
            results[i] = mlx.generate(prompt).completion
        recorded[i] = len(calls)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == list(range(8)) and all(n == 1 for n in recorded.values())


def test_compile_end_to_end(mlx, tmp_path):
    edit = "Perturb block 1's MLP output.\n```python\nw = load('" + L1 + "'); w[:, :8] += 0.5; save_edit('" + L1 + "', w)\n```"
    gen = 'Generate.\n```forward\n{"type": "generate", "model": "both", "problems": ["train_0", "train_1", "train_2"], "max_tokens": 8, "activations": {"layers": [2], "mlp_layers": [1]}}\n```'
    writer, _ = scripted_writer([edit, gen, "Final: perturbed block 1."])
    trainset = [dspy.Example(question=f"What is {i} + 2?", answer=str(i + 2)).with_inputs("question") for i in range(3)]
    opt = WeightSpace(metric=lambda ex, pred, trace=None: float(pred.answer == ex.answer), writer_lm=writer, backend=mlx,
                      max_passes=1, sandbox="none", run_dir=tmp_path / "run")
    compiled = opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    assert compiled.wsp_patch.names == [L1]
    work = tmp_path / "run" / "work"
    rows = [json.loads(line) for line in open(work / "gen_1_edited.jsonl")]
    assert [r["id"] for r in rows] == ["train_0", "train_1", "train_2"] and all(r["calls"] for r in rows)
    assert np.load(work / "gen_1_acts_original" / "mlp_mean_answer.npy").shape[1:] == (1, 64)
    expected = Patch.load(tmp_path / "run" / "patch").edited_tensors(TensorReader(mlx.weights_dir))[L1].float().numpy()
    assert np.allclose(mlx.current_param(L1), expected, atol=1e-6)

    compiled.save(str(tmp_path / "program.json"))
    loaded = dspy.Predict("question -> answer")
    loaded.load(str(tmp_path / "program.json"), allow_unsafe_lm_state=True)
    other = loaded.lm.engine.backend
    assert other is not mlx and np.allclose(other.current_param(L1), expected, atol=1e-6)
    mlx.set_weights(None)


def test_presence_penalty_skips_prompt_tokens(mlx):
    import mlx.core as mx

    from dspy_wsp.backends.mlx import _PresencePenalty

    pen = _PresencePenalty(1.5, 10)
    out = pen(mx.array([3, 4, 5]), mx.zeros((1, 10)))  # first call: the prompt only
    assert not mx.any(out).item()
    out = pen(mx.array([3, 4, 5, 7]), mx.zeros((1, 10)))
    assert out[0, 7].item() == -1.5 and out[0, 3].item() == 0.0
