"""The forward tool's decide and read passes, and the validation holdout, on the tiny d1 backend."""

import json

import dspy
import numpy as np
import pytest
from dspy.experimental import Choice

from conftest import FakeBackend, exact_answer
from decision_fixtures import build_d1
from dspy_wsp.backends.decision import D1Backend
from dspy_wsp.edits import EditStore
from dspy_wsp.forward import ForwardTool, Limits
from dspy_wsp.tensors import TensorReader

EDIT = "head.scorer.3.weight"


class Act(dspy.Signature):
    """Pick an action."""

    state: str = dspy.InputField()
    action: Choice[("stop", "Halt now."), ("continue", "Proceed."), ("review", "Ask a human.")] = dspy.OutputField(desc="What should happen?")


def metric(example, pred, trace=None):
    return float(pred.action.value == example.action)


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1f"))


@pytest.fixture(scope="module")
def backend(d1_dir):
    b = D1Backend(str(d1_dir), revision=None, framework="torch")
    yield b
    b.close()


def examples(prefix, n):
    return {f"{prefix}_{i}": dspy.Example(state=f'{{"steps": {i}, "errors": {i % 3}}}', action=["stop", "continue", "review"][i % 3]).with_inputs("state") for i in range(n)}


@pytest.fixture
def tool(backend, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    reader = TensorReader(backend.weights_dir)
    store = EditStore(work / "edited", reader, backend.default_protected())
    return ForwardTool(backend=backend, program=dspy.Predict(Act), metric=metric, practice=examples("train", 12),
                       edit_store=store, work_dir=work, limits=Limits(max_generations=20, max_generations_both=20), validation=examples("val", 9))


def _save_edit(tool, backend):
    r = TensorReader(backend.weights_dir)
    folder = tool.work_dir / "edited"
    folder.mkdir(exist_ok=True)
    np.save(folder / f"{EDIT}.npy", r.get_f32(EDIT) * -3.0 + 0.5)


def ask(tool, req, k=1):
    return tool.handle(json.dumps(req), k)


def test_decide_runs_practice_and_records_calls(tool):
    msg, counted = ask(tool, {"type": "decide", "model": "original", "problems": ["train_0", "train_1", "train_2"]})
    assert counted and '"type": "decide"' in msg
    rows = [json.loads(line) for line in open(tool.work_dir / "gen_1.jsonl")]
    assert len(rows) == 3 and all(len(r["calls"]) == 1 and "score" in r for r in rows)
    assert json.loads(rows[0]["calls"][0]["prompt"])["questions"]["action"]["type"] == "choice"


def test_validation_decide_returns_aggregates_only(tool, backend):
    _save_edit(tool, backend)
    msg, counted = ask(tool, {"type": "decide", "model": "both", "validation": True}, k=2)
    assert counted
    summary = json.loads(msg.split("```")[1])
    assert summary["validation"] is True and summary["items"] == 9 and "paired" in summary
    assert not list(tool.work_dir.glob("gen_2*"))
    assert tool.validation_log and tool.validation_log[0]["edited_tensors"] == [EDIT]
    assert "val_0" not in msg


def test_validation_and_practice_cannot_mix(tool):
    msg, counted = ask(tool, {"type": "decide", "validation": True, "problems": ["train_0"]})
    assert not counted and "validation" in msg
    msg, counted = ask(tool, {"type": "decide", "problems": ["val_0"]})
    assert not counted and "unknown practice ids" in msg


def test_read_from_recorded_pass(tool):
    ask(tool, {"type": "decide", "model": "original", "problems": ["train_3", "train_4"]}, k=3)
    msg, counted = ask(tool, {"type": "read", "model": "original", "from": {"pass": 3, "ids": ["train_4"]}, "layers": [0, 4]}, k=4)
    assert counted
    folder = tool.work_dir / "fwd_4"
    logits, probs = np.load(folder / "option_logits.npy"), np.load(folder / "probabilities.npy")
    assert logits.shape == (1, 3) and np.load(folder / "hidden_mean.npy").shape == (1, 2, 64)
    rec = [json.loads(line) for line in open(tool.work_dir / "gen_3.jsonl") if json.loads(line)["id"] == "train_4"][0]
    answer = json.loads(rec["calls"][0]["completion"])["action"]["probabilities"]
    np.testing.assert_allclose(probs[0], list(answer.values()), atol=1e-5)


def test_read_limit_enforced(tool):
    tool.limits = Limits(max_reads=2)
    reqs = [{"state": "s", "questions": {"a": {"type": "noul", "instructions": "?"}}}] * 3
    msg, counted = ask(tool, {"type": "read", "requests": reqs})
    assert not counted and "at most 2" in msg


def test_decision_backend_refuses_generate(tool):
    msg, counted = ask(tool, {"type": "generate", "problems": ["train_0"]})
    assert not counted and '"decide" or "read"' in msg


def test_llm_backend_refuses_decide_and_read(weights_dir, trainset, tmp_path):
    backend = FakeBackend(weights_dir)
    program = dspy.Predict("question -> answer")
    program.set_lm(backend.lm())
    store = EditStore(tmp_path / "edited", TensorReader(weights_dir))
    t = ForwardTool(backend=backend, program=program, metric=exact_answer, practice={f"train_{i}": ex for i, ex in enumerate(trainset)}, edit_store=store, work_dir=tmp_path)
    for kind in ("decide", "read"):
        msg, counted = t.handle(json.dumps({"type": kind, "problems": ["train_0"]}), 1)
        assert not counted and '"generate" or "teacher_forced"' in msg
