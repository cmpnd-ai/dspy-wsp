"""The Typed Decisions example's program, metric and scoring, on synthetic cases in the dataset's format."""

import math
import sys
from pathlib import Path

import dspy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import typed_decisions as td  # noqa: E402
from decision_fixtures import build_d1  # noqa: E402

QUESTIONS = {
    "action": {"type": "choice", "instructions": "What should happen?", "criteria": {"continue": "Proceed.", "stop": "Halt now.", "review": "Ask a human."}},
    "needs_review": {"type": "noul", "instructions": "This needs review.", "criteria": {"true": "A human should look.", "false": "No attention needed."}},
    "risk": {"type": "score", "instructions": "How risky?", "criteria": ["Benign.", "Low.", "Moderate."]},
}
GOLD = {
    "action": {"type": "choice", "probabilities": {"continue": 0.6, "stop": 0.1, "review": 0.3}},
    "needs_review": {"type": "noul", "probabilities": {"false": 0.7, "true": 0.3}},
    "risk": {"type": "score", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
}


def example(workflow="agent_trace", **state):
    return dspy.Example(case_id="c", workflow=workflow, state=state or {"steps": 3}, questions=QUESTIONS, gold=GOLD).with_inputs("workflow", "state")


def test_signatures_built_per_workflow():
    program = td.TypedDecisions([example("agent_trace"), example("invoices")])
    assert program.workflows == ["agent_trace", "invoices"]
    sig = program.agent_trace.signature
    assert list(sig.input_fields) == ["state"] and list(sig.output_fields) == ["action", "needs_review", "risk"]
    assert sig.output_fields["action"].json_schema_extra["desc"] == "What should happen?"


def test_metric_accuracy_and_feedback():
    pred = dspy.Prediction(
        action=dspy.experimental.Choice[tuple(QUESTIONS["action"]["criteria"].items())](value="continue", confidence=0.5, probabilities={"continue": 0.5, "stop": 0.2, "review": 0.3}),
        needs_review=dspy.experimental.Noul(value=True, confidence=0.6, probability=0.8),
        risk=dspy.experimental.Score[("Benign.", "Low.", "Moderate.")](value=1.0, confidence=0.5, probabilities={0: 0.1, 1: 0.8, 2: 0.1}),
    )
    out = td.metric(example(), pred)
    assert out.score == pytest.approx(2 / 3)
    assert "needs_review: WRONG (predicted true 0.80, gold false 0.70)" in out.feedback


def test_protocol_scoring_formulas():
    ok, kl, brier, conf = td.score_one({"continue": 0.5, "stop": 0.25, "review": 0.25}, GOLD["action"])
    assert ok and conf == 0.5
    assert kl == pytest.approx(0.6 * math.log(0.6 / 0.5) + 0.1 * math.log(0.1 / 0.25) + 0.3 * math.log(0.3 / 0.25))
    assert brier == pytest.approx(0.1**2 + 0.15**2 + 0.05**2)
    s = td.summarize([(True, 0.1, 0.2, 0.95), (False, 0.3, 0.4, 0.92), (True, 0.0, 0.0, 0.31)])
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["ece"] == pytest.approx(2 / 3 * abs(0.5 - 0.935) + 1 / 3 * abs(1 - 0.31))


def test_program_runs_on_a_decision_backend(tmp_path):
    from dspy_wsp.backends.decision import D1Backend

    backend = D1Backend(str(build_d1(tmp_path / "d1")), revision=None, framework="torch")
    try:
        program = td.TypedDecisions([example()])
        program.set_lm(backend.lm())
        pred = program(workflow="agent_trace", state={"steps": 3})
        out = td.metric(example(), pred)
        assert 0 <= out.score <= 1 and out.feedback.count(";") == 2
        noul_q = backend.lm().history[-1]["request"]["questions"]["needs_review"]
        assert noul_q["type"] == "noul" and noul_q["criteria"]["true"] == "A human should look."
    finally:
        backend.close()


def test_decided_metric_reads_decided_answers():
    import typed_decisions_reanchor as tr

    pred = dspy.Prediction(
        action=dspy.experimental.Choice[tuple(QUESTIONS["action"]["criteria"].items())](value="stop", confidence=0.5, probabilities={"continue": 0.5, "stop": 0.2, "review": 0.3}),
        needs_review=dspy.experimental.Noul(value=False, confidence=0.6, probability=0.4),
        risk=dspy.experimental.Score[("Benign.", "Low.", "Moderate.")](value=1.4, level=1, confidence=0.5, probabilities={0: 0.1, 1: 0.4, 2: 0.5}),
    )
    # decided answers: stop (gold continue) wrong, false right, level 1 right.
    # most likely options: continue right, false right, level 2 (gold 1) wrong.
    assert tr.decided_metric(example(), pred) == pytest.approx(2 / 3)
    assert td.metric(example(), pred).score == pytest.approx(2 / 3)
    assert [tr.decided(pred[q]) for q in ("action", "needs_review", "risk")] == ["stop", "false", "1"]


def test_reanchor_runs_on_a_decision_backend(tmp_path):
    import typed_decisions_reanchor as tr
    from dspy.experimental import ReAnchor

    from dspy_wsp.backends.decision import D1Backend

    backend = D1Backend(str(build_d1(tmp_path / "d1")), revision=None, framework="torch")
    try:
        cases = [example(steps=i) for i in range(12)]
        program = td.TypedDecisions(cases)
        program.set_lm(backend.lm(cache=True))
        optimizer = ReAnchor(metric=tr.decided_metric, num_threads=4)
        tuned = optimizer.compile(program, trainset=cases)
        assert optimizer.report["train_score"] >= optimizer.report["train_score_before"]
        assert tuned is not program
    finally:
        backend.close()
