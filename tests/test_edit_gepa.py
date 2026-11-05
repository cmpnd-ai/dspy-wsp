import json

import numpy as np
import pytest

from dspy_wsp.edit_gepa import EditGEPA, language

SHAPES = {"W": (3, 4), "b": (4,)}


def test_apply_statements():
    text = """
    # comments and blank lines are ignored
    W[0, :] = 2
    W[1, 1:3] = [1, -1.5]
    b[0:4:2] += 3
    W[0, 0] *= 0.5
    W[2, :] = [1, 2,
               3, 4]
    """
    w, skipped = language.apply(text, SHAPES)
    assert skipped == []
    np.testing.assert_array_equal(w["W"], [[1, 2, 2, 2], [0, 1, -1.5, 0], [1, 2, 3, 4]])
    np.testing.assert_array_equal(w["b"], [3, 0, 3, 0])


@pytest.mark.parametrize("statement", ["W[0] = 1", "X[0] = 1", "W[0, 0:2] = [1, 2, 3]", "W[0, 0] = 1e9",
                                       "import os", "W[0, 0] = foo()", "W[0, 0] /= 2"])
def test_unreadable_statements_are_skipped_not_run(statement):
    w, skipped = language.apply(f"b[0] = 1\n{statement}", SHAPES)
    assert len(skipped) == 1 and w["b"][0] == 1 and not w["W"].any()


def test_blocks_and_base():
    reply = "Explanation.\n```edits\nb[0] = 1\n```\nBetter:\n```edits\n# base: zero\nb[1] = 2\n```"
    block = language.last_block(reply)
    assert "b[1] = 2" in block and language.base_of(block, "latest") == "zero"
    assert language.last_block("no block") is None


class TargetLM:
    """A stub writer that moves one weight of b toward 1 per call."""
    history: list = []

    def __init__(self):
        self.calls = 0

    def __call__(self, messages):
        assert "THE NETWORK" in messages[-1]["content"] and "HOW IT DID" in messages[-1]["content"]
        self.calls += 1
        return [f"Raise b.\n```edits\nb[{self.calls % 4}] = 1\n```"]


def test_search_improves_and_respects_budget(tmp_path):
    def evaluate(weights, instances, with_feedback):
        total = float(weights["b"].sum())
        return [total] * len(instances), f"b sums to {total}" if with_feedback else None

    lm = TargetLM()
    search = EditGEPA(writer_lm=lm, shapes=SHAPES, evaluate=evaluate, task="TASK", budget=60, minibatch_size=2,
                      run_dir=tmp_path)
    res = search.run("W[0, 0] = 1", trainset=list(range(20)), valset=list(range(4)))
    assert res.best_val >= 2 and res.best_weights["b"].sum() == res.best_val
    assert res.units_used <= 60 and res.writer_calls == lm.calls >= 2
    assert res.best_program.startswith("W[0, 0] = 1") and "# ---- edit ----" in res.best_program
    assert json.loads((tmp_path / "result.json").read_text())["units_used"] == res.units_used
    assert len(res.candidates) >= 2 and (tmp_path / "proposals.jsonl").exists()
