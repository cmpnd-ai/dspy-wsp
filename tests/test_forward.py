import json

import dspy
import numpy as np
import pytest
from conftest import exact_answer

from dspy_wsp.edits import EditStore, save_array
from dspy_wsp.forward import ForwardTool, Limits
from dspy_wsp.tensors import TensorReader

L0 = "model.layers.0.mlp.down_proj.weight"


@pytest.fixture
def tool(backend, weights_dir, trainset, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    program = dspy.Predict("question -> answer")
    program.set_lm(backend.lm())
    store = EditStore(work / "edited", TensorReader(weights_dir))
    practice = {f"train_{i}": ex for i, ex in enumerate(trainset)}
    return ForwardTool(backend=backend, program=program, metric=exact_answer, practice=practice,
                       edit_store=store, work_dir=work, limits=Limits(max_generations=8, max_generations_both=4))


def _edit(tool):
    r = tool.edit_store.reader
    w = r.get_f32(L0)
    w[:, 1] = 0
    save_array(tool.edit_store.dir, L0, w)


def test_generate_both_reports_a_paired_gain(tool):
    _edit(tool)
    msg, counted = tool.handle(json.dumps({"type": "generate", "model": "both", "problems": ["train_0", "train_1"], "n": 2}), 1)
    assert counted, msg
    s = json.loads(msg.split("```")[1])
    assert s["original"]["score"] == 0.0 and s["edited"]["score"] == 1.0
    assert s["paired"]["gain"] == 1.0 and s["paired"]["improved"] == 4
    rows = [json.loads(line) for line in open(tool.work_dir / "gen_1_edited.jsonl")]
    assert rows[0]["calls"][0]["completion"].startswith("[[ ## answer ## ]]")
    assert rows[0]["outputs"]["answer"] == "4"


def test_generate_original_with_custom_inputs_and_activations(tool):
    req = {"type": "generate", "model": "original", "problems": ["train_0"], "inputs": [{"question": "hi?"}],
           "activations": {"layers": [0, 2], "mlp_layers": [1]}}
    msg, counted = tool.handle(json.dumps(req), 3)
    assert counted, msg
    rows = [json.loads(line) for line in open(tool.work_dir / "gen_3.jsonl")]
    assert "score" in rows[0] and "score" not in rows[1]
    assert (tool.work_dir / "gen_3_acts" / "hidden_last_prompt.npy").exists()
    assert np.load(tool.work_dir / "gen_3_acts" / "hidden_last_prompt.npy").shape == (2, 2, 4)


@pytest.mark.parametrize(
    "req, why",
    [
        ({"type": "generate", "problems": ["nope"]}, "unknown practice ids"),
        ({"type": "generate", "model": "both", "problems": ["train_0"], "n": 5}, "at most 4"),
        ({"type": "generate", "problems": []}, "give"),
        ({"type": "generate", "problems": ["train_0"], "activations": {"layers": [9]}}, "out-of-range"),
        ({"type": "generate", "inputs": [{"wrong": 1}]}, "input fields"),
        ({"type": "teacher_forced", "texts": ["a"], "chats": []}, "exactly one"),
        ({"type": "teacher_forced", "texts": ["a"], "return": {"topk": 99}}, "topk"),
        ({"type": "mystery"}, "type"),
    ],
)
def test_bad_requests_do_not_count(tool, req, why):
    msg, counted = tool.handle(json.dumps(req), 1)
    assert not counted and "not counted" in msg and why in msg


def test_invalid_json_and_invalid_edits_do_not_count(tool):
    msg, counted = tool.handle("{nope", 1)
    assert not counted and "invalid JSON" in msg
    r = tool.edit_store.reader
    save_array(tool.edit_store.dir, "model.norm.weight", r.get_f32("model.norm.weight") + 1)
    msg, counted = tool.handle(json.dumps({"type": "generate", "model": "edited", "problems": ["train_0"]}), 1)
    assert not counted and "out of scope" in msg


def test_teacher_forced_writes_arrays(tool):
    _edit(tool)
    req = {"type": "teacher_forced", "model": "edited", "texts": ["a b c", "d e"], "return": {"token_logprobs": True, "hidden_states": [1]}}
    msg, counted = tool.handle(json.dumps(req), 2)
    assert counted, msg
    s = json.loads(msg.split("```")[1])
    assert s["files"]["token_logprobs.npy"] == [2, 3] and s["mean_token_logprob"] == -1.0
    assert json.loads((tool.work_dir / "fwd_2" / "meta.json").read_text())["model"] == "edited"


def test_weights_are_restored_after_a_pass(tool, backend):
    _edit(tool)
    tool.handle(json.dumps({"type": "generate", "model": "edited", "problems": ["train_0"]}), 1)
    assert backend.current_edits == {}


def test_gepa_style_feedback_is_recorded(tool):
    tool.metric = lambda ex, pred, trace=None: dspy.Prediction(score=0.5, feedback="close, but show your work")
    msg, counted = tool.handle(json.dumps({"type": "generate", "model": "original", "problems": ["train_0"]}), 1)
    row = json.loads(open(tool.work_dir / "gen_1.jsonl").readline())
    assert row["score"] == 0.5 and row["feedback"] == "close, but show your work"
