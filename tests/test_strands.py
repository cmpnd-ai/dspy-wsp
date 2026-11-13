"""The Strands decider family on a tiny random checkpoint. MLX only."""

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from decision_fixtures import STRANDS_HEAD, STRANDS_LORA, build_strands  # noqa: E402
from dspy_wsp.backends.decision import StrandsBackend  # noqa: E402
from dspy_wsp.decision.families.strands import encoding  # noqa: E402
from dspy_wsp.decision.loading import check_decision_model, load_decision_backend, resolve_model  # noqa: E402
from dspy_wsp.decision.schema import RequestError, parse_question  # noqa: E402
from dspy_wsp.patch import Patch  # noqa: E402

STATE = {"ticket": "The invoice total is 1250 USD and the vendor is Acme."}
QUESTIONS = {
    "pick": {"type": "choice", "instructions": "Which team?", "criteria": {"technical": "Technical", "billing": "Billing", "sales": "Sales"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "level": {"type": "score", "instructions": "How urgent?", "criteria": ["level 0", "level 1", "level 2"]},
}
HEAD = ["k.bias", "k.weight", "norm.bias", "norm.weight", "q.bias", "q.weight"]


@pytest.fixture(scope="module")
def strands_dir(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DSPY_WSP_CACHE", str(tmp_path_factory.mktemp("cache")))
        yield build_strands(tmp_path_factory.mktemp("strands"))


@pytest.fixture(scope="module")
def backend(strands_dir):
    b = load_decision_backend(str(strands_dir))
    yield b
    b.close()


def test_alias_resolves_to_the_pinned_checkpoint():
    assert resolve_model("strands-2b", None) == ("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "216d1ccb46a9e9b036eb9187b248db98480f77d4")


def test_load_detects_the_family_without_a_config_json(strands_dir, backend):
    assert not (strands_dir / "config.json").exists()
    assert type(backend) is StrandsBackend and backend.framework == "mlx"


def test_torch_is_refused(strands_dir):
    with pytest.raises(NotImplementedError, match="MLX only"):
        load_decision_backend(str(strands_dir), framework="torch")


def test_the_prompt_follows_the_reference_layout():
    text, spans = encoding.render_question(parse_question(QUESTIONS["urgent"]))
    assert text == ('<question type="noul">\nDecide whether the statement is true of the state.\nIs it urgent?\n<options>\n'
                    "1. false — the statement does not hold for this state\n2. true — the statement holds for this state\n"
                    "</options>\n</question>\n<answer>")
    assert [text[a:b].split(" ")[1] for a, b in spans] == ["false", "true"]


def test_each_option_points_at_the_last_token_of_its_line(backend):
    q = parse_question(QUESTIONS["pick"])
    row = backend._encode(STATE, q)
    text, spans = encoding.render_question(q)
    offsets = backend.tok.encode(text).offsets
    assert [text[: offsets[i - row.state][1]].splitlines()[-1] for i in row.options] == [text[a:b] for a, b in spans]
    assert len(row.ids) == row.state + len(offsets)


def test_the_checkpoint_is_folded_once(strands_dir, backend):
    folded = backend.weights_dir
    assert folded != strands_dir.resolve() and not (folded / "lora").exists()
    config = json.loads((folded / "strands_decider_config.json").read_text())
    assert config["use_lora"] is False and config["folded"]["lora_weights_merged"] == len(STRANDS_LORA)
    written = (folded / "model.safetensors").stat().st_mtime_ns
    again = load_decision_backend(str(strands_dir))
    try:
        assert again.weights_dir == folded and (folded / "model.safetensors").stat().st_mtime_ns == written
    finally:
        again.close()


def test_the_adapter_is_folded_into_the_backbone(strands_dir, backend):
    from safetensors.numpy import load_file

    stem = STRANDS_LORA[0]
    base = load_file(str(strands_dir / "base" / "model.safetensors"))[f"model.language_model.{stem}.weight"]
    adapter = load_file(str(strands_dir / "lora" / "adapter_model.safetensors"))
    delta = adapter[f"base_model.model.{stem}.lora_B.weight"] @ adapter[f"base_model.model.{stem}.lora_A.weight"]
    np.testing.assert_allclose(backend.reader.get_f32(f"backbone.{stem}.weight"), base + 2.0 * delta, atol=1e-5)
    untouched = "layers.2.mlp.up_proj"
    np.testing.assert_allclose(backend.reader.get_f32(f"backbone.{untouched}.weight"),
                               load_file(str(strands_dir / "base" / "model.safetensors"))[f"model.language_model.{untouched}.weight"])


def test_the_backbone_and_the_head_take_edits(backend):
    names = backend.editable_names()
    assert set(HEAD) < set(names) and "backbone.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "backbone.layers.3.self_attn.q_proj.weight" in names and len(names) == len(backend.reader.names)
    import re

    protected = [n for n in names if any(re.search(p, n) for p in backend.default_protected())]
    assert protected == ["backbone.embed_tokens.weight", "backbone.norm.weight"]
    with pytest.raises(KeyError):
        backend.set_weights({"lm_head.weight": np.ones(4, np.float32)})


def test_answers_follow_the_request_order(backend):
    answers = backend.system_one(STATE, QUESTIONS)["answers"]
    assert list(answers) == ["pick", "urgent", "level"]
    assert list(answers["pick"]["probabilities"]) == ["technical", "billing", "sales"]
    assert sum(answers["pick"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert 0 <= answers["urgent"]["noul"] <= 1
    assert list(answers["level"]["probabilities"]) == ["0", "1", "2"]


def test_a_prompt_over_the_window_is_refused(strands_dir):
    b = load_decision_backend(str(strands_dir), max_length=40)
    try:
        with pytest.raises(RequestError, match="at most 40"):
            b.system_one(STATE, QUESTIONS)
    finally:
        b.close()


def test_a_head_edit_reuses_the_backbone_outputs(backend):
    first = backend.system_one(STATE, QUESTIONS)
    caches = backend.net._caches
    size = len(caches["original"])
    w = backend.reader.get_f32("q.weight")
    with backend.using({"q.weight": -w}):
        edited = backend.system_one(STATE, QUESTIONS)
        assert list(caches) == ["original"] and len(caches["original"]) == size
    assert edited["answers"]["pick"]["probabilities"] != first["answers"]["pick"]["probabilities"]
    assert backend.system_one(STATE, QUESTIONS) == first


def test_a_backbone_edit_gets_its_own_outputs(backend):
    first = backend.system_one(STATE, QUESTIONS)
    name = "backbone.layers.1.mlp.down_proj.weight"
    w = backend.reader.get_f32(name)
    with backend.using({name: w + 0.5}):
        edited = backend.system_one(STATE, QUESTIONS)
        assert len(backend.net._caches) == 2
    assert edited["answers"]["pick"]["probabilities"] != first["answers"]["pick"]["probabilities"]
    assert list(backend.net._caches) == ["original"]
    assert backend.system_one(STATE, QUESTIONS) == first


def test_read_gives_what_the_head_needs(backend):
    arrays, meta = backend.read([{"state": STATE, "questions": QUESTIONS}], layers=[0, 2, 4])
    assert [r["question"] for r in meta["rows"]] == ["pick", "urgent", "level"]
    assert arrays["hidden_mean"].shape == (3, 3, 64)
    get = backend.reader.get_f32

    def norm(x):
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        return x * get("norm.weight") + get("norm.bias")

    keys = norm(arrays["head_inputs"]) @ get("k.weight").T + get("k.bias")
    queries = norm(arrays["decide"]) @ get("q.weight").T + get("q.bias")
    logits = (keys * queries).sum(-1) / np.sqrt(STRANDS_HEAD["pointer_dim"])
    mask = arrays["option_mask"]
    np.testing.assert_allclose(logits[mask], arrays["option_logits"][mask], atol=1e-4)
    for row, kind in zip(range(3), ["choice", "noul", "score"]):
        z = arrays["option_logits"][row][mask[row]] / STRANDS_HEAD["temperature_by_kind"][kind]
        np.testing.assert_allclose(np.exp(z) / np.exp(z).sum(), arrays["probabilities"][row][mask[row]], atol=1e-5)


def test_check_passes(strands_dir):
    report = check_decision_model(str(strands_dir))
    assert report.ok, report.problems
    assert report.family == "strands"


def test_saved_checkpoint_reloads_with_the_edit(tmp_path, strands_dir, backend):
    w = backend.reader.get_f32("k.weight")
    mlp = "backbone.layers.2.mlp.gate_proj.weight"
    edit = {"k.weight": -3 * w + 0.5, mlp: backend.reader.get_f32(mlp) * 0.5 + 0.1}
    with backend.using(edit):
        edited = backend.system_one(STATE, QUESTIONS)
    out = Patch.from_edits(backend.reader, edit).write_checkpoint(backend.weights_dir, tmp_path / "edited")
    assert json.loads((out / "wsp_patch.json").read_text())["tensors"] == [mlp, "k.weight"]
    reloaded = load_decision_backend(str(out))
    try:
        got = reloaded.system_one(STATE, QUESTIONS)
    finally:
        reloaded.close()
    assert np.allclose(list(got["answers"]["pick"]["probabilities"].values()),
                       list(edited["answers"]["pick"]["probabilities"].values()), atol=1e-5)
