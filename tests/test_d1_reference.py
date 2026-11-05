"""Our d1 loaders against Liquid AI's reference code, on real weights.

Opt-in: set WSP_REFERENCE_TESTS=1. Downloads d1-omni-600M (about 2.3 GB) and the
Typed Decisions test split, and runs the reference model with its remote code.
"""

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("WSP_REFERENCE_TESTS") != "1", reason="set WSP_REFERENCE_TESTS=1")

REPO, REVISION = "LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674"


def typed_decisions_cases(n: int):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).slice(0, n).to_pylist()
    return [(json.loads(r["state"]), json.loads(r["questions"])) for r in rows]


@pytest.fixture(scope="module")
def snapshot():
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "*.py", "model.safetensors", "LICENSE"])


def ours(model, folder, cases):
    from dspy_wsp.decision import d1
    from dspy_wsp.decision.schema import parse_question

    tok, cfg = d1.Tokenizer(folder), d1.load_config(folder)
    rows = []
    for state, questions in cases:
        for q in questions.values():
            q = parse_question(q)
            ids, markers = d1.encode(tok, state, q, cfg["max_length"])
            rows.append((ids, markers, q))
    return [d1.probabilities(z, q, cfg["temperatures"]) for z, (_, _, q) in zip(model.logits(rows), rows)]


def test_reference_parity(snapshot):
    import torch
    from transformers import AutoModel

    from dspy_wsp.decision.d1_mlx import D1MLX
    from dspy_wsp.decision.d1_torch import D1Torch

    cases = typed_decisions_cases(20)
    ref = AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32).eval()
    expected = []
    for state, questions in cases:
        for q, p in zip(questions.values(), ref.probabilities(state, list(questions.values()))):
            expected.append(p[::-1] if q["type"] == "noul" else p)  # the reference reports a noul as [yes, no]
    del ref
    for name, model in (("torch", D1Torch.from_dir(snapshot)), ("mlx", D1MLX.from_dir(snapshot))):
        got = ours(model, snapshot, cases)
        worst = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(got, expected))
        print(f"{name}: largest probability difference {worst:.2e} over {len(expected)} questions")
        assert worst < 1e-4, name
