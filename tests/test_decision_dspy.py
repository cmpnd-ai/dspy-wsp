"""DSPy programs calling a decision backend through the TypeSafe client and the local server."""

import json

import dspy
import pytest
import requests
from dspy.experimental import Choice, Noul, Score

from decision_fixtures import build_d1
from dspy_wsp.backends.base import LOADED_BACKENDS
from dspy_wsp.backends.decision import D1Backend
from dspy_wsp.patch import Patch
from dspy_wsp.tensors import TensorReader

STATE = '{"task": "refund", "status": "open", "steps": 7}'
EDIT = "head.scorer.3.weight"


class Triage(dspy.Signature):
    """Triage an agent trace."""

    state: str = dspy.InputField()
    action: Choice[("stop", "Halt now."), ("continue", "Proceed."), ("review", "Ask a human.")] = dspy.OutputField(desc="What should happen?")
    needs_review: Noul = dspy.OutputField(desc="Does this need review?")
    risk: Score["benign", "low", "moderate", "high"] = dspy.OutputField(desc="How risky?")


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1d"))


@pytest.fixture
def backend(d1_dir):
    b = D1Backend(str(d1_dir), revision=None, framework="torch")
    yield b
    b.close()


def test_predict_with_choice_noul_score(backend):
    pred = dspy.Predict(Triage)(state=STATE, lm=backend.lm())
    assert pred.action.value in ("stop", "continue", "review")
    assert isinstance(pred.needs_review.value, bool)
    assert set(pred.risk.probabilities) == {0, 1, 2, 3}
    lm = backend.lm()
    history = lm.history[-1]["request"]
    direct = backend.system_one(history["state"], history["questions"])["answers"]
    assert pred.action.probabilities == pytest.approx(direct["action"]["probabilities"])
    assert pred.needs_review.probability == pytest.approx(direct["needs_review"]["noul"])


def test_calls_are_recorded(backend):
    with backend.record_calls() as calls:
        dspy.Predict(Triage)(state=STATE, lm=backend.lm())
    assert len(calls) == 1
    prompt = json.loads(calls[0].prompt)
    assert set(prompt["questions"]) == {"action", "needs_review", "risk"}
    assert set(json.loads(calls[0].completion)) == {"action", "needs_review", "risk"}
    assert calls[0].input_tokens > 0


def test_bad_request_is_400(backend):
    backend.lm()
    url = backend._server.url + "/v1/systemone"
    r = requests.post(url, json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"one": "only"}}}})
    assert r.status_code == 400 and "choice" in r.json()["error"]
    r = requests.post(url, json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}})
    assert r.status_code == 200 and r.json()["answers"]["q"]["type"] == "noul"


def test_save_and_load_rebuilds_backend(backend, d1_dir, tmp_path):
    r = TensorReader(backend.weights_dir)
    edits = {EDIT: r.get_f32(EDIT) * -3.0 + 0.5}
    backend.set_weights(edits)
    Patch.from_edits(r, edits, meta={"base_model": "tiny-d1"}).save(tmp_path / "patch")
    backend.patch_dir = str(tmp_path / "patch")
    program = dspy.Predict(Triage)
    program.lm = backend.lm()
    expected = program(state=STATE)
    program.save(str(tmp_path / "program.json"))
    LOADED_BACKENDS.clear()
    loaded = dspy.Predict(Triage)
    loaded.load(str(tmp_path / "program.json"), allow_unsafe_lm_state=True)
    assert loaded.lm._backend is not backend and loaded.lm.base_url != backend.lm().base_url
    got = loaded(state=STATE)
    assert got.action.probabilities == pytest.approx(expected.action.probabilities, abs=1e-5)
    loaded.lm._backend.close()


def test_cached_client_keys_on_weights(backend):
    r = TensorReader(backend.weights_dir)
    lm = backend.lm(cache=True)
    assert lm.cache and backend.lm(cache=True) is lm and backend.lm() is not lm
    program = dspy.Predict(Triage)
    program.lm = lm
    before = program(state=STATE).action.probabilities
    with backend.using({EDIT: r.get_f32(EDIT) * -3.0 + 0.5}):
        edited = program(state=STATE).action.probabilities
        assert backend.weights_fingerprint != "original"
    again = program(state=STATE).action.probabilities
    assert edited != before and again == before
