import json

import numpy as np
import pytest

from decision_fixtures import build_d1, build_nanojev
from dspy_wsp.backends.decision import D1Backend, NanoJevBackend
from dspy_wsp.decision.loading import (
    DecisionModelError,
    check_decision_model,
    load_decision_backend,
    resolve_model,
)
from dspy_wsp.patch import Patch
from dspy_wsp.tensors import TensorReader

STATE = {"ticket": "The invoice total is 1250 USD and the vendor is Acme."}
QUESTIONS = {
    "pick": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "technical": "Technical"}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
}


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def nj_dir(tmp_path_factory):
    return build_nanojev(tmp_path_factory.mktemp("nanojev"))


def test_aliases_resolve_to_pinned_checkpoints():
    assert resolve_model("d1", None) == ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674")
    assert resolve_model("nanojev", None) == ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a")
    assert resolve_model("C-Tianyu/NanoJev", None)[1] == "047b927b30882a1138fc504821b82ac145a4b81a"
    assert resolve_model("C-Tianyu/NanoJev", "main") == ("C-Tianyu/NanoJev", "main")
    assert resolve_model("someone/other-model", None) == ("someone/other-model", None)


@pytest.mark.parametrize("which,cls", [("d1", D1Backend), ("nanojev", NanoJevBackend)])
def test_load_detects_family(which, cls, d1_dir, nj_dir):
    folder = d1_dir if which == "d1" else nj_dir
    b = load_decision_backend(str(folder), framework="torch", max_length=512)
    try:
        assert type(b) is cls
        assert b.system_one(STATE, QUESTIONS)["answers"]["urgent"]["type"] == "noul"
    finally:
        b.close()


def test_unknown_architecture_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(DecisionModelError, match="not a supported decision model"):
        load_decision_backend(str(tmp_path))


def test_missing_folder_is_refused(tmp_path):
    with pytest.raises(DecisionModelError, match="no config.json"):
        load_decision_backend(str(tmp_path / "missing"))


@pytest.mark.parametrize("framework", ["torch", "mlx"])
@pytest.mark.parametrize("which", ["d1_omni", "nanojev"])
def test_check_passes_on_good_checkpoints(which, framework, d1_dir, nj_dir):
    if framework == "mlx":
        pytest.importorskip("mlx.core")
    report = check_decision_model(str(d1_dir if which == "d1_omni" else nj_dir), framework=framework, max_length=512)
    assert report.ok, report.problems
    assert report.family == which
    assert {"load", "tensors", "answers", "batching", "edits"} <= {c["name"] for c in report.checks}
    assert "PASS" in report.summary()


def test_check_reports_a_checkpoint_that_does_not_load(tmp_path, d1_dir):
    import shutil

    from safetensors.torch import load_file, save_file

    broken = tmp_path / "broken"
    shutil.copytree(d1_dir, broken)
    weights = load_file(str(broken / "model.safetensors"))
    del weights["head.scorer.3.weight"]
    save_file(weights, str(broken / "model.safetensors"))
    report = check_decision_model(str(broken), framework="torch")
    assert not report.ok
    assert report.family == "d1_omni"
    assert any("load" in p for p in report.problems)
    assert "FAIL" in report.summary()


def test_check_reports_an_unknown_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    report = check_decision_model(str(tmp_path))
    assert not report.ok and report.family is None


def test_saved_nanojev_checkpoint_reloads_with_the_edit(tmp_path, nj_dir):
    b = load_decision_backend(str(nj_dir), framework="torch", max_length=512)
    try:
        w = TensorReader(b.weights_dir).get_f32("scalar.weight")
        edit = {"scalar.weight": -3 * w + 0.5}
        with b.using(edit):
            edited = b.system_one(STATE, QUESTIONS)
        out = Patch.from_edits(b.reader, edit).write_checkpoint(b.weights_dir, tmp_path / "edited")
    finally:
        b.close()
    assert (out / "backbone_config" / "config.json").is_file()
    assert (out / "tokenizer" / "tokenizer.json").is_file()
    assert (out / "wsp_patch.json").is_file()
    reloaded = load_decision_backend(str(out), framework="torch", max_length=512)
    try:
        assert type(reloaded) is NanoJevBackend
        got = reloaded.system_one(STATE, QUESTIONS)
    finally:
        reloaded.close()
    assert got["answers"]["urgent"]["noul"] == pytest.approx(edited["answers"]["urgent"]["noul"], abs=1e-5)
    assert np.allclose(list(got["answers"]["pick"]["probabilities"].values()),
                       list(edited["answers"]["pick"]["probabilities"].values()), atol=1e-5)


def test_cli_check(d1_dir, tmp_path, capsys):
    from dspy_wsp.cli import main

    assert main(["check", str(d1_dir), "--framework", "torch"]) == 0
    assert "PASS" in capsys.readouterr().out
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert main(["check", str(tmp_path)]) == 1
