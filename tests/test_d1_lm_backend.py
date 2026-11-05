import numpy as np
import pytest

from decision_fixtures import build_d1_lm
from dspy_wsp.decision.loading import check_decision_model, load_decision_backend
from dspy_wsp.decision.schema import RequestError
from dspy_wsp.patch import Patch
from dspy_wsp.tensors import TensorReader

STATE = {"ticket": "Customer was charged twice for the March plan.", "plan": "team"}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "tech": "Tech", "sales": "Sales"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "Needs action today", "false": "Can wait"}},
    "severity": {"type": "score", "instructions": "How bad?", "criteria": ["minor", "major", "critical"]},
}
EDIT = "model.language_model.layers.5.feed_forward.w2.weight"


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return build_d1_lm(tmp_path_factory.mktemp("d1lmb"))


@pytest.fixture(scope="module", params=["torch", "mlx"])
def backend(request, folder):
    if request.param == "mlx":
        pytest.importorskip("mlx.core")
    b = load_decision_backend(str(folder), framework=request.param)
    yield b
    b.close()


def test_family_and_answers(backend):
    from dspy_wsp.backends.decision import D1LMBackend

    assert type(backend) is D1LMBackend
    answers = backend.system_one(STATE, QUESTIONS)["answers"]
    assert answers["team"]["choice"] in QUESTIONS["team"]["criteria"]
    assert 0 <= answers["urgent"]["noul"] <= 1
    assert set(answers["severity"]["probabilities"]) == {"0", "1", "2"}


def test_seventeen_questions(backend):
    qs = {f"q{i}": {"type": "noul", "instructions": f"Is fact {i} stated?"} for i in range(17)}
    answers = backend.system_one(STATE, qs)["answers"]
    assert list(answers) == list(qs)
    assert all(sum(a["probabilities"].values()) == pytest.approx(1.0) for a in answers.values())


@pytest.mark.parametrize("raw", [
    {"type": "text", "instructions": "Who?"},
    {"type": "choice", "multiple": True, "instructions": "Which?", "criteria": {"a": "A", "b": "B"}},
])
def test_unsupported_questions_are_refused(backend, raw):
    with pytest.raises(RequestError, match="choice, noul and score"):
        backend.system_one(STATE, {"x": raw})


def test_read_matches_the_readout(backend):
    arrays, meta = backend.read([{"state": STATE, "questions": QUESTIONS}], layers=[0, 6])
    assert [r["question"] for r in meta["rows"]] == list(QUESTIONS)
    assert arrays["head_inputs"].shape == arrays["option_rows"].shape == (3, 3, 64)
    assert arrays["hidden_mean"].shape == (3, 2, 64)
    dots = np.nansum(arrays["head_inputs"] * arrays["option_rows"], axis=-1)
    mask = arrays["option_mask"]
    assert np.allclose(dots[mask], arrays["option_logits"][mask], atol=1e-4)


def test_edit_moves_and_restores(backend):
    before = backend.system_one(STATE, QUESTIONS)
    w = TensorReader(backend.weights_dir).get_f32(EDIT)
    with backend.using({EDIT: -2 * w + 0.3}):
        edited = backend.system_one(STATE, QUESTIONS)
    assert edited["answers"]["urgent"]["noul"] != pytest.approx(before["answers"]["urgent"]["noul"])
    assert backend.system_one(STATE, QUESTIONS) == before
    assert "model.language_model.embed_tokens.weight" in backend.editable_names()
    assert any("embed_tokens" in p for p in backend.default_protected())


def test_check_and_save(folder, tmp_path):
    report = check_decision_model(str(folder), framework="torch")
    assert report.ok, report.problems
    assert report.family == "d1_lm"
    b = load_decision_backend(str(folder), framework="torch")
    try:
        w = TensorReader(b.weights_dir).get_f32(EDIT)
        edit = {EDIT: -2 * w + 0.3}
        with b.using(edit):
            edited = b.system_one(STATE, QUESTIONS)
        out = Patch.from_edits(b.reader, edit).write_checkpoint(b.weights_dir, tmp_path / "edited")
    finally:
        b.close()
    r = load_decision_backend(str(out), framework="torch")
    try:
        got = r.system_one(STATE, QUESTIONS)
    finally:
        r.close()
    assert got["answers"]["urgent"]["noul"] == pytest.approx(edited["answers"]["urgent"]["noul"], abs=1e-5)


def test_alias():
    from dspy_wsp.decision.loading import ALIASES

    assert ALIASES["d1-2b"] == ("LiquidAI/d1-2B", "652f920e544ea37f81e4b0ce89b81d4ed0da90b1")


def test_describe_names_the_norm_after_the_last_mlp(folder):
    b = load_decision_backend(str(folder), framework="torch")
    try:
        text = b.describe()
    finally:
        b.close()
    assert "writes straight into" not in text
    assert "layers.5.post_ffn_norm" in text


def test_chunks_bound_the_attention_masks(folder):
    from dspy_wsp.decision.families.d1_lm import encoding as enc
    from dspy_wsp.decision.families.d1_lm.encoding import MASK_BUDGET
    from dspy_wsp.decision.families.d1_lm.torch import D1LMTorch

    net = D1LMTorch.__new__(D1LMTorch)
    width = 2000
    rows = [enc.Row(list(range(width)), 10, [-1] * width, [width - 1], [], []) for _ in range(8)]
    sizes = [len(c) for c in net._chunks(rows)]
    assert sum(sizes) == 8
    assert all(n == 1 or n * width * width <= MASK_BUDGET for n in sizes)


def test_backend_applies_per_type_temperatures(folder, tmp_path):
    import json
    import shutil

    from dspy_wsp.decision.families.d1_lm import encoding as enc

    hot = tmp_path / "hot"
    shutil.copytree(folder, hot)
    (hot / "calibration.json").write_text(json.dumps({"default": 1.337, "temperatures": {"noul": 4.0}}))
    q = {"type": "noul", "instructions": "Is it urgent?"}
    b = load_decision_backend(str(hot), framework="torch")
    try:
        arrays, _ = b.read([{"state": STATE, "questions": {"u": q}}])
        p_true = b.system_one(STATE, {"u": q})["answers"]["u"]["noul"]
    finally:
        b.close()
    assert p_true == pytest.approx(enc.probabilities(arrays["option_logits"][0], 4.0)[1], abs=1e-6)
