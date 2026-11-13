"""The Clef family on a tiny random checkpoint. MLX only."""

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from decision_fixtures import build_clef  # noqa: E402
from dspy_wsp.backends.decision import ClefBackend  # noqa: E402
from dspy_wsp.decision.loading import check_decision_model, load_decision_backend, resolve_model  # noqa: E402
from dspy_wsp.patch import Patch  # noqa: E402
from dspy_wsp.tensors import TensorReader  # noqa: E402

STATE = {"ticket": "The invoice total is 1250 USD and the vendor is Acme."}
QUESTIONS = {
    "pick": {"type": "choice", "instructions": "Which team?", "criteria": {"technical": "Technical", "billing": "Billing", "sales": "Sales"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "level": {"type": "score", "instructions": "How urgent?", "criteria": ["level 0", "level 1", "level 2"]},
}


@pytest.fixture(scope="module")
def clef_dir(tmp_path_factory):
    return build_clef(tmp_path_factory.mktemp("clef"))


@pytest.fixture(scope="module")
def backend(clef_dir):
    b = load_decision_backend(str(clef_dir), max_length=4096)
    yield b
    b.close()


def test_alias_resolves_to_the_pinned_checkpoint():
    assert resolve_model("clef-flash", None) == ("Cloudflare/clef-flash", "22b75507981345858dd152fbd9a6f8c0c4068b0a")


def test_index_covers_the_shards_and_the_head_file(clef_dir):
    reader = TensorReader(clef_dir)
    assert reader.index["residual_scorer.3.weight"] == "joint_head.safetensors"
    assert reader.index["lm_head.weight"] == "model-00001-of-00001.safetensors"


def test_load_detects_the_family(backend):
    assert type(backend) is ClefBackend and backend.framework == "mlx"


def test_torch_is_refused(clef_dir):
    with pytest.raises(NotImplementedError, match="MLX only"):
        load_decision_backend(str(clef_dir), framework="torch")


def test_only_the_head_takes_edits(backend):
    names = backend.editable_names()
    assert "residual_scorer.0.weight" in names and "evidence_layers.0.attention.in_proj_weight" in names
    assert not [n for n in names if n.startswith(("model.", "lm_head"))]
    with pytest.raises(KeyError):
        backend.set_weights({"lm_head.weight": backend.reader.get_f32("lm_head.weight")})


def test_answers_follow_the_request_order(backend):
    answers = backend.system_one(STATE, QUESTIONS)["answers"]
    assert list(answers) == ["pick", "urgent", "level"]
    assert list(answers["pick"]["probabilities"]) == ["technical", "billing", "sales"]
    assert sum(answers["pick"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert 0 <= answers["urgent"]["noul"] <= 1
    assert list(answers["level"]["probabilities"]) == ["0", "1", "2"]


def test_a_choice_does_not_depend_on_the_order_of_its_criteria(backend):
    flipped = {**QUESTIONS, "pick": {**QUESTIONS["pick"], "criteria": dict(reversed(QUESTIONS["pick"]["criteria"].items()))}}
    a = backend.system_one(STATE, QUESTIONS)["answers"]["pick"]["probabilities"]
    b = backend.system_one(STATE, flipped)["answers"]["pick"]["probabilities"]
    assert {k: pytest.approx(v, abs=1e-6) for k, v in a.items()} == b


def test_backbone_outputs_are_reused(backend):
    backend.system_one(STATE, QUESTIONS)
    cache = backend.net._backbone._cache
    size = len(cache)
    first = backend.system_one(STATE, QUESTIONS)
    w = backend.reader.get_f32("residual_scorer.3.weight")
    with backend.using({"residual_scorer.3.weight": -w}):
        edited = backend.system_one(STATE, QUESTIONS)
    assert len(cache) == size
    assert edited["answers"]["pick"]["probabilities"] != first["answers"]["pick"]["probabilities"]
    assert backend.system_one(STATE, QUESTIONS) == first


def test_read_gives_what_the_scorer_needs(backend):
    arrays, meta = backend.read([{"state": STATE, "questions": QUESTIONS}], layers=[0, 2, 4])
    assert [r["question"] for r in meta["rows"]] == ["pick", "urgent", "level"]
    assert arrays["hidden_mean"].shape == (3, 3, 64)
    get = backend.reader.get_f32
    x = arrays["head_inputs"] @ get("residual_scorer.0.weight").T + get("residual_scorer.0.bias")
    from math import erf, sqrt

    gelu = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / sqrt(2))))
    residual = (gelu(x) @ get("residual_scorer.3.weight").T + get("residual_scorer.3.bias"))[..., 0]
    gate = 1 / (1 + np.exp(-get("residual_gate")))
    logits = arrays["prior"] + gate * (np.exp(get("joint_logit_scale")) * arrays["cosine"] + residual)
    mask = arrays["option_mask"]
    np.testing.assert_allclose(logits[mask], arrays["option_logits"][mask], atol=1e-4)


def test_check_passes(clef_dir):
    report = check_decision_model(str(clef_dir), max_length=4096)
    assert report.ok, report.problems
    assert report.family == "clef"


def test_saved_checkpoint_reloads_with_the_edit(tmp_path, clef_dir, backend):
    w = backend.reader.get_f32("residual_scorer.3.weight")
    edit = {"residual_scorer.3.weight": -3 * w + 0.5}
    with backend.using(edit):
        edited = backend.system_one(STATE, QUESTIONS)
    out = Patch.from_edits(backend.reader, edit).write_checkpoint(backend.weights_dir, tmp_path / "edited")
    assert json.loads((out / "wsp_patch.json").read_text())["tensors"] == ["residual_scorer.3.weight"]
    reloaded = load_decision_backend(str(out), max_length=4096)
    try:
        got = reloaded.system_one(STATE, QUESTIONS)
    finally:
        reloaded.close()
    assert np.allclose(list(got["answers"]["pick"]["probabilities"].values()),
                       list(edited["answers"]["pick"]["probabilities"].values()), atol=1e-5)
