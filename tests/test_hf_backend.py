"""The Hugging Face backend on a tiny random Llama (no downloads)."""

import json

import dspy
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from conftest import scripted_writer  # noqa: E402

from dspy_wsp import HFBackend, Patch, WeightSpace  # noqa: E402
from dspy_wsp.tensors import TensorReader  # noqa: E402

L1 = "model.layers.1.mlp.down_proj.weight"


@pytest.fixture(scope="module")
def hf(tiny_model):
    return HFBackend(str(tiny_model), device="cpu", max_tokens=12, attn_implementation="eager")


def test_describe_and_names(hf):
    assert "2 layers" in hf.describe() and "width 32" in hf.describe()
    assert L1 in hf.editable_names()
    assert hf.dims()["intermediate"] == 64


def test_seeded_generation_through_dspy(hf):
    lm = hf.lm()
    hf.seed(7)
    a = lm("hello")[0]
    hf.seed(7)
    b = lm("hello")[0]
    hf.seed(8)
    c = lm("hello")[0]
    assert a == b and isinstance(a, str)
    assert a != c or len(a) == 0
    with hf.record_calls() as calls:
        lm("hello")
    assert calls[0].output_tokens <= 12 and calls[0].prompt.endswith("<s>assistant\n")


def _big_edit(hf):
    r = TensorReader(hf.weights_dir)
    g = np.random.default_rng(0)
    return {L1: r.get_f32(L1) + g.normal(0, 1.0, size=r.shape(L1)).astype(np.float32)}


def test_teacher_forced_matches_full_logits_and_sees_edits(hf):
    texts = ["What is 2 + 2?", "Respond with answer"]
    arrays, meta = hf.teacher_forced(texts, token_logprobs=True, topk=3, hidden_states=[0, 2], mlp_act=[1], attn_layers=[0])
    T = int(arrays["lengths"].max())
    assert arrays["hidden_states"].shape == (2, 2, T, 32)
    assert arrays["mlp_act"].shape == (2, 1, T, 64)
    assert arrays["attentions"].shape == (2, 1, 4, T, T)
    ids = hf.tokenizer(texts[0], add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        logits = hf.model(torch.tensor([ids])).logits[0].float()
    ref = torch.log_softmax(logits, -1)[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0].numpy()
    assert np.allclose(arrays["token_logprobs"][0, 1 : len(ids)], ref, atol=1e-3)
    assert np.isnan(arrays["token_logprobs"][0, 0])
    with hf.using(_big_edit(hf)):
        edited, _ = hf.teacher_forced(texts[:1], token_logprobs=True)
    assert not np.allclose(edited["token_logprobs"][0, 1 : len(ids)], ref, atol=1e-3)
    again, _ = hf.teacher_forced(texts[:1], token_logprobs=True)
    assert np.allclose(again["token_logprobs"][0, 1 : len(ids)], ref, atol=1e-3)  # restored


def test_pooled_activations(hf):
    out = hf.pooled_activations([("abc", "def"), ("question", "")], layers=[1], mlp_layers=[0, 1])
    assert out["hidden_last_prompt"].shape == (2, 1, 32)
    assert out["mlp_mean_answer"].shape == (2, 2, 64)
    assert np.isnan(out["hidden_mean_answer"][1]).all()  # no answer tokens


def test_patch_checkpoint_loads_in_transformers(hf, tmp_path):
    edits = _big_edit(hf)
    patch = Patch.from_edits(TensorReader(hf.weights_dir), edits)
    out = patch.write_checkpoint(hf.weights_dir, tmp_path / "edited")
    m = transformers.AutoModelForCausalLM.from_pretrained(str(out))
    got = dict(m.named_parameters())[L1].detach()
    assert torch.equal(got, torch.from_numpy(edits[L1]).to(torch.bfloat16))


def test_compile_end_to_end(hf, tiny_model, tmp_path):
    edit = "Perturb block 1's MLP output.\n```python\nw = load('" + L1 + "'); w[:, :8] += 0.5; save_edit('" + L1 + "', w)\n```"
    tf = 'Read two texts with both models.\n```forward\n{"type": "teacher_forced", "model": "edited", "texts": ["What is 2 + 2?"], "return": {"token_logprobs": true, "mlp_act": [1]}}\n```'
    gen = 'Generate.\n```forward\n{"type": "generate", "model": "both", "problems": ["train_0", "train_1"], "max_tokens": 8, "activations": {"layers": [2], "mlp_layers": [1]}}\n```'
    writer, engine = scripted_writer([edit, tf, gen, "Final: perturbed block 1."])
    trainset = [dspy.Example(question=f"What is {i} + 2?", answer=str(i + 2)).with_inputs("question") for i in range(3)]
    opt = WeightSpace(metric=lambda ex, pred, trace=None: float(pred.answer == ex.answer), writer_lm=writer, backend=hf,
                      max_passes=2, sandbox="none", run_dir=tmp_path / "run")
    compiled = opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    assert compiled.wsp_patch.names == [L1]
    assert compiled.wsp_report["budget"]["passes_used"] == 2
    work = tmp_path / "run" / "work"
    rows = [json.loads(line) for line in open(work / "gen_2_edited.jsonl")]
    assert rows[0]["calls"] and rows[0]["score"] == 0.0  # a random model fails, and is scored
    assert np.load(work / "gen_2_acts_original" / "mlp_mean_answer.npy").shape[1:] == (1, 64)
    assert np.load(work / "fwd_1" / "mlp_act.npy").shape[1] == 1
    # the backend now runs the edited weights
    p = dict(hf.model.named_parameters())[L1].detach()
    expected = Patch.load(tmp_path / "run" / "patch").edited_tensors(TensorReader(hf.weights_dir))[L1]
    assert torch.equal(p, expected)

    # a saved program reloads with the base model and the patch applied
    compiled.save(str(tmp_path / "program.json"))
    loaded = dspy.Predict("question -> answer")
    loaded.load(str(tmp_path / "program.json"), allow_unsafe_lm_state=True)
    other = loaded.lm.engine.backend if hasattr(loaded.lm, "engine") else None
    if other is None:
        from dspy_wsp.backends.hf import _LOADED

        other = next(iter(_LOADED.values()))
    assert other is not hf
    assert torch.equal(dict(other.model.named_parameters())[L1].detach(), expected)
    hf.set_weights(None)


def test_vision_language_checkpoint_names_map_to_text_params(tiny_model, tmp_path):
    import shutil

    from safetensors.torch import load_file, save_file

    d = tmp_path / "vl"
    shutil.copytree(tiny_model, d)
    tensors = load_file(str(d / "model.safetensors"))
    renamed = {k.replace("model.", "model.language_model.", 1): v for k, v in tensors.items()}
    renamed["model.visual.proj.weight"] = torch.zeros(2, 2)
    save_file(renamed, str(d / "model.safetensors"), metadata={"format": "pt"})
    vl = HFBackend(str(d), device="cpu", max_tokens=4, attn_implementation="eager")
    name = "model.language_model.layers.1.mlp.down_proj.weight"
    assert name in vl.editable_names() and "model.visual.proj.weight" not in vl.editable_names()
    w = TensorReader(d).get_f32(name) + 1.0
    with vl.using({name: w}):
        assert torch.equal(dict(vl.model.named_parameters())[L1].detach(), torch.from_numpy(w).to(torch.bfloat16))
