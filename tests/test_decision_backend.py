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


def test_mlx_cache_stays_under_limit(d1_dir):
    mx = pytest.importorskip("mlx.core")
    from dspy_wsp.backends.decision import MLX_CACHE_LIMIT

    b = D1Backend(str(d1_dir), revision=None, framework="mlx")
    try:
        for i in range(20):
            b.system_one({"steps": i, "note": "word " * (i * 7)}, QUESTIONS)
        assert b._call(mx.get_cache_memory) <= MLX_CACHE_LIMIT
    finally:
        b.close()


class _Packed(D1Backend):
    """Every question of a request in one row, to exercise request-level encoding."""

    def _encode_request(self, state, questions):
        return [([self._encode(state, q) for q in questions.values()], list(questions))]

    def _row_tokens(self, row):
        return sum(len(r[0]) for r in row)

    def _row_outputs(self, output, n):
        assert len(output) == n
        return output

    def _load_network(self):
        net = super()._load_network()

        class Loop:
            def logits(self, rows):
                return [net.logits(r) for r in rows]

            def capture(self, rows, layers):
                return [net.capture(r, layers) for r in rows]

            def __getattr__(self, name):
                return getattr(net, name)

        return Loop()


def test_request_rows_can_hold_several_questions(d1_dir):
    plain = D1Backend(str(d1_dir), revision=None, framework="torch")
    packed = _Packed(str(d1_dir), revision=None, framework="torch")
    try:
        assert packed.system_one(STATE, QUESTIONS)["answers"] == plain.system_one(STATE, QUESTIONS)["answers"]
        a, meta = packed.read([{"state": STATE, "questions": QUESTIONS}])
        b, _ = plain.read([{"state": STATE, "questions": QUESTIONS}])
        assert [r["question"] for r in meta["rows"]] == list(QUESTIONS)
        assert np.allclose(a["option_logits"], b["option_logits"], equal_nan=True)
    finally:
        plain.close()
        packed.close()


def test_an_error_after_encoding_fails_the_call_and_keeps_the_worker(d1_dir):
    from concurrent.futures import ThreadPoolExecutor

    class Broken(D1Backend):
        def _row_outputs(self, output, n):
            raise ValueError("broken readout")

    b = Broken(str(d1_dir), revision=None, framework="torch")
    pool = ThreadPoolExecutor(1)
    call = pool.submit(b.system_one, STATE, QUESTIONS)
    try:
        with pytest.raises(ValueError, match="broken readout"):
            call.result(timeout=20)
        assert b._worker.is_alive()
    finally:
        pool.shutdown(wait=False)
        if b._worker.is_alive():
            b.close()
