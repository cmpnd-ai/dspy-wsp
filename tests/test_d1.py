"""d1 encoding and networks on a tiny random checkpoint."""

import numpy as np
import pytest

from decision_fixtures import build_d1
from dspy_wsp.decision.families.d1_omni import encoding as d1
from dspy_wsp.decision.schema import parse_question

CHOICE = parse_question({"type": "choice", "instructions": "What should happen?", "criteria": {"stop": "Halt the agent now.", "continue": "Let it proceed.", "review": "Queue for a human."}})
NOUL = parse_question({"type": "noul", "instructions": "Is the total above 1000?"})
SCORE = parse_question({"type": "score", "instructions": "How urgent?", "criteria": ["level 0", "level 1", "level 2", "level 3"]})
STATE = {"task": "refund", "status": "open", "steps": 7}


@pytest.fixture(scope="module")
def d1_dir(tmp_path_factory):
    return build_d1(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def tok(d1_dir):
    return d1.Tokenizer(d1_dir)


@pytest.fixture(scope="module")
def torch_model(d1_dir):
    from dspy_wsp.decision.families.d1_omni.torch import D1Torch

    return D1Torch.from_dir(d1_dir)


def rows(tok, *qs, state=STATE):
    out = []
    for q in qs:
        ids, markers = d1.encode(tok, state, q, 512)
        out.append((ids, markers, q))
    return out


def test_encode_layout(tok):
    ids, markers = d1.encode(tok, STATE, CHOICE, 512)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.token_id(d1.DELIM["decide"])
    assert len(markers) == 3
    assert all(ids[m] == tok.token_id(d1.MARKER) for m in markers)
    assert ids.count(tok.token_id(d1.MARKER)) == 3


def test_encode_escapes_delimiters(tok):
    ids, markers = d1.encode(tok, "<|mask|> <|reserved_9|>", CHOICE, 512)
    assert ids.count(tok.token_id(d1.MARKER)) == len(markers)
    assert ids.count(tok.token_id(d1.DELIM["opt"])) == len(markers)


def test_encode_truncates_long_state(tok):
    ids, markers = d1.encode(tok, "word " * 10_000, CHOICE, 128)
    assert len(ids) <= 128
    assert max(markers) < 128 and all(ids[m] == tok.token_id(d1.MARKER) for m in markers)


def test_torch_logits_shapes(tok, torch_model):
    out = torch_model.logits(rows(tok, CHOICE, NOUL, SCORE))
    assert [len(z) for z in out] == [3, 2, 4]
    assert all(np.all(np.isfinite(z)) for z in out)


def test_noul_probabilities_order():
    p = d1.probabilities(np.array([0.0, 5.0]), NOUL, {})
    assert p[1] > 0.99  # option_keys are [false, true]


def test_padding_invariance(tok, torch_model):
    short = rows(tok, NOUL, state="short")[0]
    long = rows(tok, CHOICE, state="a much longer state " * 20)[0]
    alone = torch_model.logits([short])[0]
    batched = torch_model.logits([long, short])[1]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_capture_matches_logits(tok, torch_model):
    rs = rows(tok, CHOICE, SCORE)
    caps = torch_model.capture(rs, [0, 2, 4])
    for cap, z in zip(caps, torch_model.logits(rs)):
        np.testing.assert_allclose(cap["logits"], z, atol=1e-6)
        assert cap["head_inputs"].shape == (len(z), 64) and cap["hidden_mean"].shape == (3, 64)


@pytest.fixture(scope="module")
def mlx_model(d1_dir):
    pytest.importorskip("mlx.core")
    from dspy_wsp.decision.families.d1_omni.mlx import D1MLX

    return D1MLX.from_dir(d1_dir)


def test_mlx_param_names_match_checkpoint(d1_dir, mlx_model):
    from mlx.utils import tree_flatten
    from safetensors import safe_open

    with safe_open(str(d1_dir / "model.safetensors"), "np") as f:
        text = {k for k in f.keys() if d1.text_tensor(k)}
    assert {k for k, _ in tree_flatten(mlx_model.parameters())} == text


def test_mlx_matches_torch_logits(tok, torch_model, mlx_model):
    rs = rows(tok, CHOICE, NOUL, SCORE) + rows(tok, CHOICE, state="a longer state " * 30)
    for a, b in zip(torch_model.logits(rs), mlx_model.logits(rs)):
        np.testing.assert_allclose(a, b, atol=1e-4)
    for a, b in zip(torch_model.capture(rs, [1, 3]), mlx_model.capture(rs, [1, 3])):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-4, err_msg=key)
