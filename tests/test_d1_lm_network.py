import numpy as np
import pytest
import torch

from decision_fixtures import build_d1_lm, d1_lm_config
from dspy_wsp.decision.families.d1_lm import encoding as enc
from dspy_wsp.decision.families.d1_lm.lfm3_torch import LanguageModel
from dspy_wsp.decision.families.d1_lm.torch import D1LMTorch
from dspy_wsp.decision.schema import parse_question

STATE = {"ticket": "Customer was charged twice for the March plan.", "plan": "team"}
QS = [
    parse_question({"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Billing", "tech": "Tech", "sales": "Sales"}}),
    parse_question({"type": "noul", "instructions": "Is it urgent?"}),
    parse_question({"type": "score", "instructions": "How bad?", "criteria": ["minor", "major", "critical"]}),
]


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return build_d1_lm(tmp_path_factory.mktemp("d1lm"))


@pytest.fixture(scope="module")
def tok(folder):
    return enc.Tokenizer(folder)


@pytest.fixture(scope="module")
def net(folder, tok):
    return D1LMTorch.from_dir(folder, pad_token_id=tok.pad_token_id)


def rows_for(tok, state, qs):
    return enc.encode_request(tok, state, qs, max_length=4096)


def test_loads_with_checkpoint_names(folder, net):
    from safetensors import safe_open

    with safe_open(str(folder / "model.safetensors"), framework="pt") as f:
        names = {k for k in f.keys() if k.startswith("model.language_model.")}
    assert set(dict(net.named_parameters())) == names


def test_logits_per_question(tok, net):
    out = net.logits(rows_for(tok, STATE, QS))
    assert len(out) == 1 and [len(z) for z in out[0]] == [3, 2, 3]


def test_padding_does_not_change_answers(tok, net):
    short = rows_for(tok, "Short.", QS[:1])
    long = rows_for(tok, {"ticket": "word " * 200}, QS)
    alone = net.logits(short)[0][0]
    batched = net.logits(short + long)[0][0]
    assert np.allclose(alone, batched, atol=1e-5)


def test_state_reads_the_questions(tok, net):
    other = parse_question({"type": "noul", "instructions": "Did the customer mention a refund at any point?"})
    a = net.logits(rows_for(tok, STATE, [QS[0], QS[1]]))[0][0]
    b = net.logits(rows_for(tok, STATE, [QS[0], other]))[0][0]
    assert np.abs(a - b).max() > 1e-6


def test_capture_shapes_and_readout(tok, net):
    caps = net.capture(rows_for(tok, STATE, QS), [0, 3])[0]
    for cap, q in zip(caps, QS):
        k = len(q.option_keys)
        assert cap["head_inputs"].shape == (k, 64) and cap["option_rows"].shape == (k, 64)
        assert cap["hidden_mean"].shape == (2, 64)
        assert np.allclose((cap["head_inputs"] * cap["option_rows"]).sum(-1), cap["logits"], atol=1e-4)


def _causal(t):
    m = torch.tril(torch.ones(t, t, dtype=torch.bool))[None]
    return m, m, torch.ones(1, t, dtype=torch.bool)


def test_conv_is_causal(tok):
    torch.manual_seed(0)
    lm = LanguageModel(d1_lm_config(400)["text_config"]).eval()
    ids = torch.randint(0, 400, (1, 12))
    changed = ids.clone()
    changed[0, -1] = (ids[0, -1] + 1) % 400
    with torch.no_grad():
        a, _ = lm(ids, *_causal(12))
        b, _ = lm(changed, *_causal(12))
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6)


def test_sliding_window_limits_reach():
    cfg = dict(d1_lm_config(400)["text_config"], layer_types=["sliding_attention", "sliding_attention"], num_hidden_layers=2,
               sliding_window=4)
    torch.manual_seed(0)
    lm = LanguageModel(cfg).eval()
    ids = torch.randint(0, 400, (1, 12))
    changed = ids.clone()
    changed[0, 0] = (ids[0, 0] + 1) % 400
    rows = [enc.Row(list(range(12)), 0, [1] * 12, [11], [], [])]
    full, sliding, real = (torch.from_numpy(m) for m in enc.masks(rows, 12, 4))
    with torch.no_grad():
        a, _ = lm(ids, full, sliding, real)
        b, _ = lm(changed, full, sliding, real)
    assert torch.allclose(a[0, 10], b[0, 10], atol=1e-6)
    assert not torch.allclose(a[0, 1], b[0, 1], atol=1e-6)


@pytest.fixture(scope="module")
def mlx_net(folder, tok):
    pytest.importorskip("mlx.core")
    from dspy_wsp.decision.families.d1_lm.mlx import D1LMMLX

    return D1LMMLX.from_dir(folder, pad_token_id=tok.pad_token_id)


def _requests(tok):
    many = [parse_question({"type": "noul", "instructions": f"Is fact {i} stated?"}) for i in range(17)]
    return {
        "one": rows_for(tok, STATE, QS[:1]),
        "seventeen": rows_for(tok, STATE, many),
        "two_lengths": rows_for(tok, "Short.", QS) + rows_for(tok, {"ticket": "word " * 150}, QS),
    }


@pytest.mark.parametrize("case", ["one", "seventeen", "two_lengths"])
def test_mlx_matches_torch(case, tok, net, mlx_net):
    rows = _requests(tok)[case]
    a = net.capture(rows, [0, 3, 6])
    b = mlx_net.capture(rows, [0, 3, 6])
    for ra, rb in zip(a, b):
        for qa, qb in zip(ra, rb):
            for key in ("logits", "head_inputs", "option_rows", "hidden_mean"):
                assert np.allclose(qa[key], qb[key], atol=1e-4), key


def test_mlx_parameter_names(net, mlx_net):
    from mlx.utils import tree_flatten

    assert {k for k, _ in tree_flatten(mlx_net.parameters())} == set(dict(net.named_parameters()))


def test_query_chunks_change_nothing(tok, net, mlx_net, monkeypatch):
    from dspy_wsp.decision.families.d1_lm import lfm3_mlx, lfm3_torch

    rows = _requests(tok)["two_lengths"]
    whole_torch, whole_mlx = net.logits(rows), mlx_net.logits(rows)
    monkeypatch.setattr(lfm3_torch, "QUERY_CHUNK", 7)
    monkeypatch.setattr(lfm3_mlx, "QUERY_CHUNK", 5)
    for whole, part in ((whole_torch, net.logits(rows)), (whole_mlx, mlx_net.logits(rows))):
        for a, b in zip(whole, part):
            for qa, qb in zip(a, b):
                assert np.allclose(qa, qb, atol=1e-5)
