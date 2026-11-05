"""NanoJev encoding, networks and backend on a tiny random checkpoint."""

import numpy as np
import pytest

from decision_fixtures import build_nanojev
from dspy_wsp.decision import nanojev
from dspy_wsp.decision.schema import RequestError, parse_question

CHOICE = parse_question({"type": "choice", "instructions": "What should happen?", "criteria": {"stop": "Halt now.", "continue": "Proceed.", "review": "Ask a human."}})
NOUL = parse_question({"type": "noul", "instructions": "Is the total above 1000?", "criteria": {"true": "It is.", "false": "It is not."}})
SCORE = parse_question({"type": "score", "instructions": "How urgent?", "criteria": ["level 0", "level 1", "level 2"]})
STATE = {"task": "refund", "status": "open"}


@pytest.fixture(scope="module")
def nj_dir(tmp_path_factory):
    return build_nanojev(tmp_path_factory.mktemp("nj"))


@pytest.fixture(scope="module")
def tok(nj_dir):
    return nanojev.Tokenizer(nj_dir / "tokenizer")


@pytest.fixture(scope="module")
def torch_model(nj_dir, tok):
    from dspy_wsp.decision.nanojev_torch import NanoJevTorch

    return NanoJevTorch.from_dir(nj_dir, pad_token_id=tok.pad_token_id)


def rows(tok, *qs, state=STATE):
    return [(*nanojev.encode(tok, state, q, 512), q) for q in qs]


def test_encode_matches_reference_layout(tok):
    paths, prefix = nanojev.encode(tok, STATE, CHOICE, 512)
    assert len(paths) == 3 and all(p[-1] == tok.eos_token_id for p in paths)
    assert all(p[:prefix] == paths[0][:prefix] for p in paths)
    text = tok._tok.decode(paths[1], skip_special_tokens=False)
    assert text.startswith("State:\n{'task': 'refund', 'status': 'open'}\nQuestion type: choice\nQuestion:\nWhat should happen?\n")
    assert text.endswith("Candidate:\ncontinue: Proceed.\nDecision:<|im_end|>")
    noul_paths, _ = nanojev.encode(tok, "s", NOUL, 512)
    noul_text = tok._tok.decode(noul_paths[0], skip_special_tokens=False)
    assert len(noul_paths) == 1 and "Question type: boolean" in noul_text and "False criterion: It is not.\nTrue criterion: It is.\n" in noul_text
    assert "Candidate:\nThe proposition is true." in noul_text
    score_text = tok._tok.decode(nanojev.encode(tok, "s", SCORE, 512)[0][2], skip_special_tokens=False)
    assert "Candidate:\nlevel 2\nDecision:" in score_text


def test_long_path_raises(tok):
    with pytest.raises(RequestError):
        nanojev.encode(tok, "word " * 2000, CHOICE, 128)


def test_boolean_logits_zero_and_z(tok, torch_model):
    z = torch_model.logits(rows(tok, NOUL))[0]
    assert z.shape == (2,) and z[0] == 0.0


def test_logits_shapes_and_padding(tok, torch_model):
    out = torch_model.logits(rows(tok, CHOICE, NOUL, SCORE))
    assert [len(z) for z in out] == [3, 2, 3]
    alone = torch_model.logits(rows(tok, SCORE, state="x"))[0]
    batched = torch_model.logits(rows(tok, CHOICE, state="a much longer state " * 20) + rows(tok, SCORE, state="x"))[1]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_mlx_matches_torch(nj_dir, tok, torch_model):
    pytest.importorskip("mlx.core")
    from dspy_wsp.decision.nanojev_mlx import NanoJevMLX

    mlx_model = NanoJevMLX.from_dir(nj_dir, pad_token_id=tok.pad_token_id)
    rs = rows(tok, CHOICE, NOUL, SCORE) + rows(tok, CHOICE, state="another, longer state " * 10)
    for a, b in zip(torch_model.capture(rs, [0, 1, 2]), mlx_model.capture(rs, [0, 1, 2])):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-4, err_msg=key)


@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_backend_edit_scalar_changes_answers(nj_dir, framework):
    if framework == "mlx":
        pytest.importorskip("mlx.core")
    from dspy_wsp.backends.decision import NanoJevBackend
    from dspy_wsp.tensors import TensorReader

    b = NanoJevBackend(str(nj_dir), revision=None, framework=framework, max_length=512)
    try:
        qs = {"a": {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}}, "b": {"type": "noul", "instructions": "?"}}
        before = b.system_one(STATE, qs)
        w = TensorReader(b.weights_dir).get_f32("scalar.weight")
        with b.using({"scalar.weight": -3 * w + 0.5}):
            edited = b.system_one(STATE, qs)
        assert edited["answers"]["b"]["noul"] != pytest.approx(before["answers"]["b"]["noul"])
        assert b.system_one(STATE, qs) == before
        assert "norm.weight" in b.editable_names() and not any(p.endswith(r"(^|\.)norm\.weight$") for p in b.default_protected())
    finally:
        b.close()


def test_choice_without_descriptions_uses_keys(tok):
    q = parse_question({"type": "choice", "instructions": "Who owns it?", "criteria": {"alice": None, "bob": ""}})
    text = tok._tok.decode(nanojev.encode(tok, "s", q, 512)[0][0], skip_special_tokens=False)
    assert "Candidate:\nalice\nDecision:" in text
