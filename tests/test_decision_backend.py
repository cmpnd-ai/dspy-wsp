"""DecisionBackend with the tiny d1 checkpoint, in both frameworks."""

import threading

import numpy as np
import pytest

from decision_fixtures import build_d1
from dspy_wsp.backends.decision import D1Backend
from dspy_wsp.tensors import TensorReader

QUESTIONS = {
    "action": {"type": "choice", "instructions": "What should happen?", "criteria": {"stop": "Halt now.", "continue": "Proceed.", "review": "Ask a human."}},
    "review": {"type": "noul", "instructions": "Does this need review?"},
    "risk": {"type": "score", "instructions": "How risky?", "criteria": ["benign", "low", "moderate", "high"]},
}
STATE = {"task": "refund", "status": "open", "steps": 7}
EDIT = "head.scorer.3.weight"


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1b"))


@pytest.fixture(scope="module", params=["torch", "mlx"])
def backend(request, d1_dir):
    if request.param == "mlx":
        pytest.importorskip("mlx.core")
    b = D1Backend(str(d1_dir), revision=None, framework=request.param)
    yield b
    b.close()


def _edit(backend):
    r = TensorReader(backend.weights_dir)
    return {EDIT: r.get_f32(EDIT) * -3.0 + 0.5}


def test_system_one_answers_all_types(backend):
    out = backend.system_one(STATE, QUESTIONS)
    a = out["answers"]
    assert a["action"]["choice"] in QUESTIONS["action"]["criteria"]
    assert 0 <= a["review"]["noul"] <= 1
    assert set(a["risk"]["probabilities"]) == {"0", "1", "2", "3"}
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0


def test_edit_changes_and_restores(backend):
    before = backend.system_one(STATE, QUESTIONS)
    with backend.using(_edit(backend)):
        edited = backend.system_one(STATE, QUESTIONS)
    after = backend.system_one(STATE, QUESTIONS)
    assert edited["answers"]["action"]["probabilities"] != before["answers"]["action"]["probabilities"]
    assert after == before


def test_swap_waits_for_running_requests(backend):
    original = backend.system_one(STATE, QUESTIONS)["answers"]
    with backend.using(_edit(backend)):
        edited = backend.system_one(STATE, QUESTIONS)["answers"]
    results, stop = [], threading.Event()

    def ask():
        while not stop.is_set():
            results.append(backend.system_one(STATE, QUESTIONS)["answers"])

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for _ in range(5):
        backend.set_weights(_edit(backend))
        backend.set_weights(None)
    stop.set()
    for t in threads:
        t.join()
    # Batched answers can differ from single ones in the last bits; mixed weights would differ by far more.
    gaps = [min(_gap(r, original), _gap(r, edited)) for r in results]
    assert results and max(gaps) < 1e-4, max(gaps)


def _gap(a, b):
    def flat(ans):
        out = []
        for v in ans.values():
            out.extend(v["probabilities"].values() if "probabilities" in v else [v["noul"]])
        return np.array(out)

    return float(np.max(np.abs(flat(a) - flat(b))))


def test_read_shapes_and_logits_match_answers(backend):
    arrays, meta = backend.read([{"state": STATE, "questions": QUESTIONS}], layers=[0, 2])
    answers = backend.system_one(STATE, QUESTIONS)["answers"]
    assert arrays["option_logits"].shape == (3, 4) and arrays["head_inputs"].shape == (3, 4, 64)
    assert arrays["hidden_mean"].shape == (3, 2, 64)
    assert [r["question"] for r in meta["rows"]] == list(QUESTIONS)
    np.testing.assert_allclose(arrays["probabilities"][0, :3], list(answers["action"]["probabilities"].values()), atol=1e-5)
    assert arrays["probabilities"][1, 1] == pytest.approx(answers["review"]["noul"], abs=1e-5)
    assert not arrays["option_mask"][0, 3] and np.isnan(arrays["option_logits"][0, 3])


def test_editable_names_exclude_media_and_include_head(backend):
    names = backend.editable_names()
    assert EDIT in names and "encoder.layers.0.conv.in_proj.weight" in names
    assert not any(n.startswith(("vision.", "audio.")) for n in names)
    assert backend.dims()["layers"] == 4


def test_bad_request_fails_alone(backend):
    from dspy_wsp.decision.schema import RequestError

    with pytest.raises(RequestError):
        backend.system_one(STATE, {"x": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}})
    assert backend.system_one(STATE, QUESTIONS)["answers"]
