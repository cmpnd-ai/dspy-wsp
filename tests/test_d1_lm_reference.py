"""Our d1-2B code against Liquid AI's own, on the real weights.

Opt-in: set WSP_REFERENCE_TESTS=1. Downloads the pinned d1-2B checkpoint (about
3.8 GB) with its model code, and the Typed Decisions test split. Liquid's code
runs through transformers' remote code, on the CPU in float32; it is freed
before our model loads.
"""

import gc
import importlib
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("WSP_REFERENCE_TESTS") != "1", reason="set WSP_REFERENCE_TESTS=1")

REPO, REVISION = "LiquidAI/d1-2B", "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
ENCODING_CASES, ANSWER_CASES = 100, 30


def _requests(n):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from typed_decisions import load_split

    out = []
    for ex in load_split("test"):
        qs = {name: q for name, q in ex.questions.items() if q["type"] in ("choice", "noul", "score") and not q.get("multiple")}
        if qs:
            out.append((ex.state, qs))
        if len(out) == n:
            break
    return out


@pytest.fixture(scope="module")
def reference():
    """Liquid's sequences for ENCODING_CASES requests and its answers for the first ANSWER_CASES."""
    import torch
    from transformers import AutoModel

    requests = _requests(ENCODING_CASES)
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True, dtype=torch.float32).eval()
    prompt = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".prompt")
    engine = model.engine
    sequences = []
    for state, qs in requests:
        seqs = engine._sequences([(state, [prompt.as_question(q) for q in qs.values()], [])], set_size=16)
        sequences.append([(ids, state_len, question, [e[-1] for e in ends]) for _, _, ids, state_len, question, ends, _ in seqs])
    answers = [model.system_one(state, qs)["answers"] for state, qs in requests[:ANSWER_CASES]]
    del engine, model
    gc.collect()
    return requests, sequences, answers


def _distribution(answer: dict, q: dict) -> list[float]:
    if q["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return [float(v) for v in answer["probabilities"].values()]


def test_encoding_matches_liquid(reference):
    from huggingface_hub import snapshot_download

    from dspy_wsp.decision.families.d1_lm import encoding as enc
    from dspy_wsp.decision.schema import parse_question

    requests, sequences, _ = reference
    tok = enc.Tokenizer(snapshot_download(REPO, revision=REVISION, allow_patterns=["tokenizer*.json"]))
    for (state, qs), theirs in zip(requests, sequences):
        ours = enc.encode_request(tok, state, [parse_question(q) for q in qs.values()], max_length=131_072, scheme=2)
        assert len(ours) == len(theirs)
        for row, (ids, state_len, question, slots) in zip(ours, theirs):
            assert row.ids == list(ids)
            assert row.state_len == state_len
            assert row.question_of == list(question)
            assert row.slots == list(slots)


@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_answers_match_liquid(reference, framework):
    if framework == "mlx":
        pytest.importorskip("mlx.core")
    from dspy_wsp.backends.decision import D1LMBackend

    requests, _, answers = reference
    backend = D1LMBackend(framework=framework)
    try:
        worst = 0.0
        for (state, qs), theirs in zip(requests[:ANSWER_CASES], answers):
            ours = backend.system_one(state, qs)["answers"]
            for name, q in qs.items():
                gap = np.abs(np.array(_distribution(ours[name], q)) - np.array(_distribution(theirs[name], q))).max()
                worst = max(worst, float(gap))
        assert worst < 1e-4, worst
    finally:
        backend.close()
