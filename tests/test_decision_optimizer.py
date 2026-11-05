"""WeightSpace with a decision backend: the decision prompt, the validation holdout and a scripted run."""

import json

import dspy
import pytest
from dspy.experimental import Choice

from conftest import scripted_writer
from decision_fixtures import build_d1
from dspy_wsp import WeightSpace
from dspy_wsp.backends.decision import D1Backend

EDIT_NAME = "head.scorer.3.weight"
ACTIONS = ["stop", "continue", "review"]


class Act(dspy.Signature):
    """Pick an action."""

    state: str = dspy.InputField()
    action: Choice[("stop", "Halt now."), ("continue", "Proceed."), ("review", "Ask a human.")] = dspy.OutputField(desc="What should happen?")


def metric(example, pred, trace=None):
    return float(pred.action.value == example.action)


def trainset(n=40):
    return [dspy.Example(workflow=f"w{i % 4}", state=f'{{"steps": {i}}}', action=ACTIONS[i % 3]).with_inputs("workflow", "state") for i in range(n)]


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.act = dspy.Predict(Act)

    def forward(self, workflow, state):
        return self.act(state=state)


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1o"))


@pytest.fixture
def backend(d1_dir):
    b = D1Backend(str(d1_dir), revision=None, framework="torch")
    yield b
    b.close()


EXPLAIN_EDIT = ("Flip the scorer's output weights.\n```python\n# hypothesis: flipping the scorer changes answers\n"
                f"w = load('{EDIT_NAME}'); save_edit('{EDIT_NAME}', -2 * w + 0.1)\n```")
VALIDATE = 'Check the edit on the validation set.\n```forward\n{"type": "decide", "model": "both", "validation": true}\n```'
DECIDE = 'Compare both models on two items.\n```forward\n{"type": "decide", "model": "both", "problems": ["train_0", "train_1"]}\n```'
FINAL = "I flipped the scorer. Validation confirmed nothing in particular."


def make(backend, tmp_path, replies, **kw):
    writer, engine = scripted_writer(replies)
    opts = dict(max_passes=2, sandbox="none", run_dir=tmp_path / "run", kernel_timeout=60, holdout=0.25, holdout_key=lambda ex: ex.workflow)
    opts.update(kw)
    return WeightSpace(metric=metric, writer_lm=writer, backend=backend, **opts), engine


def test_compile_runs_with_scripted_writer(backend, tmp_path):
    opt, engine = make(backend, tmp_path, [EXPLAIN_EDIT, DECIDE, VALIDATE, FINAL])
    valset = [dspy.Example(workflow="w0", state='{"held": "out"}', action="stop").with_inputs("workflow", "state")]
    compiled = opt.compile(Program(), trainset=trainset(), valset=valset)
    report = compiled.wsp_report
    assert report["validation_size"] == 10 and len(report["validation_runs"]) == 1
    assert report["validation_runs"][0]["edited_tensors"] == [EDIT_NAME]
    assert report["validation"]["paired"]["items"] == 1
    task = (tmp_path / "run" / "task_prompt.txt").read_text()
    assert '"type": "decide"' in task and '"type": "read"' in task and "10 examples is held out" in task
    assert "head.scorer" in task and "vision" not in task
    assert "you may recompute the decision head" in task


def test_holdout_split_is_stratified_and_hidden(backend, tmp_path):
    opt, _ = make(backend, tmp_path, [DECIDE, DECIDE, FINAL])
    opt.compile(Program(), trainset=trainset())
    rows = [json.loads(line) for line in open(tmp_path / "run" / "work" / "practice" / "trainset.jsonl")]
    assert len(rows) == 30
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["validation_size"] == 10
    per_workflow = {}
    for r in rows:
        per_workflow[r["inputs"]["workflow"]] = per_workflow.get(r["inputs"]["workflow"], 0) + 1
    assert sorted(per_workflow.values()) == [7, 7, 8, 8]


def test_backend_default_protected_used(backend):
    opt = WeightSpace(metric=metric, writer_lm=None, backend=backend)
    assert opt.protected == backend.default_protected()
    assert not any(p.endswith(r"(^|\.)norm\.weight$") for p in opt.protected)


def test_holdout_refused_for_llm_backends(weights_dir):
    from conftest import FakeBackend

    with pytest.raises(ValueError):
        WeightSpace(metric=metric, writer_lm=None, backend=FakeBackend(weights_dir), holdout=5)
