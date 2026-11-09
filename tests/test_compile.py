"""End to end: WeightSpace.compile with a scripted writer and the fake backend."""

import json

import dspy
import pytest
from conftest import exact_answer, scripted_writer

from dspy_wsp import Patch, WeightSpace

L0 = "model.layers.0.mlp.down_proj.weight"

EXPLORE = "Check the tensors.\n```python\n# hypothesis: names load\nlen(NAMES), SHAPES['" + L0 + "']\n```"
EDIT = "Zero one input column of block 0's MLP output.\n```python\nw = load('" + L0 + "'); w[:, 1] = 0; save_edit('" + L0 + "', w)\n```"
MEASURE = 'Compare both models on three problems.\n```forward\n{"type": "generate", "model": "both", "problems": ["train_0", "train_1", "train_2"]}\n```'
FINAL = "I zeroed column 1 of block 0's MLP output projection. Paired gain 1.0 on 3 problems."


def make(backend, tmp_path, replies, **kw):
    writer, engine = scripted_writer(replies)
    opts = dict(max_passes=1, sandbox="none", run_dir=tmp_path / "run", kernel_timeout=30)
    opts.update(kw)
    return WeightSpace(metric=exact_answer, writer_lm=writer, backend=backend, **opts), engine


def test_compile_returns_a_program_on_the_edited_weights(backend, trainset, tmp_path):
    opt, engine = make(backend, tmp_path, [EXPLORE, EDIT, MEASURE, FINAL])
    student = dspy.Predict("question -> answer")
    valset = [dspy.Example(question="Held out: 9 - 5?", answer="4").with_inputs("question")]
    compiled = opt.compile(student, trainset=trainset, valset=valset)

    assert compiled(question="What is 3 + 1?").answer == "4"
    assert compiled.wsp_patch.names == [L0]
    report = compiled.wsp_report
    assert report["budget"]["passes_used"] == 1 and report["budget"]["code_runs_used"] == 2
    assert report["validation"]["paired"]["gain"] == 1.0
    assert report["final_answer"] == FINAL
    run = tmp_path / "run"
    assert Patch.load(run / "patch").names == [L0]
    assert (run / "turns" / "t001" / "prompt.txt").read_text().startswith("TASK.")
    assert "OUTPUT of Python run 1 of at most 300" in engine.received[1][-1][1]
    # the task prompt describes the program and the metric, and marks protected tensors
    task = (run / "task_prompt.txt").read_text()
    assert "question -> answer" in task and "exact_answer" in task and "(out of scope)" in task
    # the valset never reached the writer's folder
    assert "Held out" not in (run / "work" / "practice" / "trainset.jsonl").read_text()
    assert student.lm is None  # the student was not modified


def test_early_final_answers_are_nudged_and_unexplained_blocks_sent_back(backend, trainset, tmp_path):
    bare = "```python\n1 + 1\n```"
    opt, engine = make(backend, tmp_path, ["I'm done.", bare, bare, bare, MEASURE, FINAL])
    opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    msgs = [m[-1][1] for m in engine.received]
    assert "continues until all 1 forward passes" in msgs[1]
    assert "was not run" in msgs[2] and "was not run" in msgs[3]
    assert "OUTPUT of Python run 1" in msgs[4]  # third bare block runs, as in the paper
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["budget"]["passes_used"] == 1


def test_no_edit_means_an_empty_patch(backend, trainset, tmp_path):
    opt, _ = make(backend, tmp_path, [MEASURE, FINAL])
    compiled = opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    assert len(compiled.wsp_patch) == 0
    assert compiled(question="1+3?").answer == "5"


def test_fallback_if_worse(backend, trainset, tmp_path):
    def prefers_five(example, pred, trace=None):
        return float(pred.answer == "5")

    writer, _ = scripted_writer([EDIT, MEASURE, FINAL])
    opt = WeightSpace(metric=prefers_five, writer_lm=writer, backend=backend, max_passes=1, sandbox="none",
                      run_dir=tmp_path / "run", fallback_if_worse=True)
    valset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]
    compiled = opt.compile(dspy.Predict("question -> answer"), trainset=trainset, valset=valset)
    assert len(compiled.wsp_patch) == 0 and "fallback" in compiled.wsp_report


def test_handoff_restarts_the_conversation(backend, trainset, tmp_path):
    notes = "Keep notes.\n```python\nopen(WORK + '/notes.md', 'w').write('best: none yet')\n```"
    opt, engine = make(backend, tmp_path, [notes, EXPLORE, MEASURE, FINAL], handoff_every=2)
    opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    third = engine.received[2]
    assert len(third) == 1 and third[0][1].startswith("HANDOFF.")
    assert "best: none yet" in third[0][1]


def test_resume_finished_run_returns_the_same_answer(backend, trainset, tmp_path):
    opt, _ = make(backend, tmp_path, [EDIT, MEASURE, FINAL])
    opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    opt2, engine2 = make(backend, tmp_path, [], resume=True)
    compiled = opt2.compile(dspy.Predict("question -> answer"), trainset=trainset)
    assert engine2.received == []  # nothing left to do
    assert compiled.wsp_report["final_answer"] == FINAL
    assert compiled.wsp_patch.names == [L0]


def test_trainset_needs_inputs(backend, tmp_path):
    opt, _ = make(backend, tmp_path, [])
    with pytest.raises(ValueError, match="with_inputs"):
        opt.compile(dspy.Predict("question -> answer"), trainset=[dspy.Example(question="q", answer="a")])


def test_each_turn_records_the_writers_usage(backend, trainset, tmp_path):
    opt, _ = make(backend, tmp_path, [EXPLORE, EDIT, MEASURE, FINAL])
    opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    run = tmp_path / "run"
    keys = {"cache_hit", "tokens_in", "cached_in", "tokens_out", "reasoning_tokens", "usd"}
    turns = [json.loads((run / "turns" / f"t{i:03d}" / "usage.json").read_text()) for i in range(1, 5)]
    assert all(set(t) == keys for t in turns)
    total = json.loads((run / "state.json").read_text())["writer_usage"]
    assert total["calls"] == 4
    assert total["tokens_in"] == sum(t["tokens_in"] for t in turns)


def test_compile_runs_the_writer_in_the_sandbox(backend, trainset, tmp_path):
    from dspy_wsp.kernel import KernelError, find_sandbox

    try:
        find_sandbox("auto")
    except KernelError:
        pytest.skip("no sandbox on this machine")
    opt, _ = make(backend, tmp_path, [EXPLORE, EDIT, MEASURE, FINAL], sandbox="auto")
    compiled = opt.compile(dspy.Predict("question -> answer"), trainset=trainset)
    assert compiled.wsp_patch.names == [L0]
    assert "no internet" in (tmp_path / "run" / "task_prompt.txt").read_text()
