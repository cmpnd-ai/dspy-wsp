"""WeightSpace with GEPA's population search: settings, the round flow, the budget and the logs."""

import json

import dspy
import numpy as np
import pytest

from conftest import FakeBackend, scripted_writer
from decision_fixtures import build_d1
from dspy_wsp import GEPASearch, WeightSpace
from dspy_wsp.backends.decision import D1Backend
from dspy_wsp.search import restore_candidate
from test_decision_optimizer import EDIT_NAME, Program, trainset

EDIT = ("Flip the scorer's output weights.\n```python\n# hypothesis: flipping the scorer changes answers\n"
        f"w = load('{EDIT_NAME}'); save_edit('{EDIT_NAME}', -2 * w + 0.1)\n```")
DECIDE = 'Compare both models on two items.\n```forward\n{"type": "decide", "model": "both", "problems": ["train_0", "train_1"]}\n```'
VALIDATE = 'Check the edit on the validation set.\n```forward\n{"type": "decide", "model": "both", "validation": true}\n```'
FINAL = "I flipped the scorer."


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1g"))


@pytest.fixture
def backend(d1_dir):
    b = D1Backend(str(d1_dir), revision=None, framework="torch")
    yield b
    b.close()


def edited_metric(backend):
    """Scores 1 for any edited model and 0 for the original, so acceptance is predictable."""
    return lambda example, pred, trace=None: float(bool(backend.current_edits))


def make(backend, tmp_path, replies, **search):
    writer, engine = scripted_writer(replies)
    settings = dict(min_rounds=2, minibatch_size=2, passes_per_proposal=1, code_runs_per_proposal=5) | search
    s = GEPASearch(**settings)
    opt = WeightSpace(metric=edited_metric(backend), writer_lm=writer, backend=backend, max_passes=s.min_passes(),
                      sandbox="none", run_dir=tmp_path / "run", kernel_timeout=60, holdout=0.25,
                      holdout_key=lambda ex: ex.workflow, search=s)
    return opt, engine


def valset():
    return [dspy.Example(workflow="w0", state='{"held": "out"}', action="stop").with_inputs("workflow", "state")]


def test_gepa_settings_and_checks(backend, weights_dir):
    s = GEPASearch()
    assert s.round_cost == 5 and s.min_passes() == 51
    metric = edited_metric(backend)
    with pytest.raises(ValueError, match="holdout"):
        WeightSpace(metric=metric, writer_lm=None, backend=backend, max_passes=51, search=s)
    with pytest.raises(ValueError, match="51"):
        WeightSpace(metric=metric, writer_lm=None, backend=backend, max_passes=50, holdout=10, search=s)
    with pytest.raises(ValueError, match="decision"):
        WeightSpace(metric=metric, writer_lm=None, backend=FakeBackend(weights_dir), max_passes=51, search=s)


def test_gepa_round_flow(backend, tmp_path):
    # round 1: a refused validation pass, an edit, a counted decide pass, then done; round 2: done at once
    opt, engine = make(backend, tmp_path, [VALIDATE, EDIT, DECIDE, FINAL, FINAL])
    compiled = opt.compile(Program(), trainset=trainset(), valset=valset())
    report, run = compiled.wsp_report, tmp_path / "run"

    assert report["search"]["rounds"] == 2 and report["search"]["best"] == "c001"
    assert compiled.wsp_patch.names == [EDIT_NAME]
    assert report["validation"]["paired"]["items"] == 1

    # c002 started from c001 and changed nothing, so it inherits the edit and is rejected
    assert (run / "candidates" / "c002" / "edited" / f"{EDIT_NAME}.npy").exists()
    assert not list((run / "candidates" / "c000" / "edited").glob("*.npy"))

    # seed validation 1, then round 1: parent 1, writer 1, child 1, validation 1; round 2: parent 1, child 1
    assert report["budget"]["passes_used"] == 7
    assert (run / "work" / "gen_3_original.jsonl").exists()  # the writer's pass is numbered across the run

    first = (run / "rounds" / "r01" / "turns" / "t001" / "prompt.txt").read_text()
    assert "ROUND 1" in first and "c000" in first and "VALIDATION is not available" in first
    assert '"validation": true' not in first
    transcript = (run / "rounds" / "r01" / "transcript.jsonl").read_text()
    assert "FAILED (not counted)" in transcript
    second = engine.received[-1][0][1]
    assert "ROUND 2" in second and "c001" in second and EDIT_NAME in second


def test_gepa_logs(backend, tmp_path):
    opt, _ = make(backend, tmp_path, [EDIT, FINAL, FINAL])
    opt.compile(Program(), trainset=trainset())
    run = tmp_path / "run"
    rows = [json.loads(line) for line in open(run / "rounds.jsonl")]
    assert [r["round"] for r in rows][:2] == [1, 2]
    assert rows[0]["parent"] == "c000" and rows[0]["child"] == "c001"
    assert rows[0]["accepted"] and rows[0]["child_val"] == 1.0 and rows[0]["best_val"] == 1.0
    assert rows[0]["parent_minibatch"] == 0.0 and rows[0]["child_minibatch"] == 1.0
    assert not rows[1]["accepted"] and rows[1]["parent"] == "c001" and rows[1]["child_val"] is None
    assert all(r["passes_used"] <= opt.max_passes for r in rows)
    progress = (run / "progress.md").read_text()
    assert "| 1 | c000 | c001 |" in progress and "| 2 | c001 | c002 |" in progress


def test_restore_candidate_replaces_the_edit_folder(tmp_path):
    cand, work = tmp_path / "c001", tmp_path / "work"
    (cand / "edited").mkdir(parents=True)
    np.save(cand / "edited" / "a.npy", np.ones(2, np.float32))
    (cand / "notes.md").write_text("parent notes")
    (work / "edited").mkdir(parents=True)
    np.save(work / "edited" / "b.npy", np.zeros(2, np.float32))
    (work / "notes.md").write_text("child notes")
    restore_candidate(cand, work)
    assert sorted(p.name for p in (work / "edited").glob("*.npy")) == ["a.npy"]
    assert (work / "notes.md").read_text() == "parent notes"


def test_restore_candidate_without_notes_clears_them(tmp_path):
    cand, work = tmp_path / "c000", tmp_path / "work"
    (cand / "edited").mkdir(parents=True)
    work.mkdir()
    (work / "notes.md").write_text("child notes")
    restore_candidate(cand, work)
    assert not (work / "notes.md").exists() and (work / "edited").is_dir()
