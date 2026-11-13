"""Our Clef encoding and joint schema head against Clef's own `joint_schema_model.py`.

Opt-in: set WSP_REFERENCE_TESTS=1. Downloads Clef-Flash's tokenizer, head and model
code (about 265 MB), not its backbone. Both heads read the same random hidden
states, so the test runs without the 9B backbone.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("WSP_REFERENCE_TESTS") != "1", reason="set WSP_REFERENCE_TESTS=1")

REPO, REVISION = "Cloudflare/clef-flash", "22b75507981345858dd152fbd9a6f8c0c4068b0a"
FILES = ["joint_schema_model.py", "joint_head.safetensors", "joint_head_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja"]
STATE = {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}, "note": "Second reminder sent."}
QUESTIONS = {
    "status": {"type": "choice", "instructions": "What is the invoice status?",
               "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
    "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    "strict": {"type": "noul", "instructions": "Is the vendor Acme?", "criteria": {"true": "The vendor is Acme.", "false": "Another vendor."}},
    "urgency": {"type": "score", "instructions": "", "criteria": ["Can wait", "This week", "Today"]},
}


@pytest.fixture(scope="module")
def folder():
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, allow_patterns=FILES)


@pytest.fixture(scope="module")
def reference(folder):
    spec = importlib.util.spec_from_file_location("joint_schema_model", f"{folder}/joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # its dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


def ours_and_theirs(folder, reference, state=STATE, questions=QUESTIONS):
    from transformers import AutoTokenizer

    from dspy_wsp.decision.families.clef import encoding
    from dspy_wsp.decision.schema import parse_question

    parsed = {name: parse_question(q) for name, q in questions.items()}
    row = encoding.encode_request(encoding.Tokenizer(folder), state, parsed)
    record = reference.encode_record(AutoTokenizer.from_pretrained(folder), {"state": state, "questions": questions})
    return parsed, row, record


def test_encoding_matches_the_reference(folder, reference):
    parsed, row, record = ours_and_theirs(folder, reference)
    assert row.ids == record.input_ids
    for (name, q), field, theirs in zip(parsed.items(), row.fields, record.questions):
        assert theirs.question_id == name and field.type_id == theirs.question_type
        assert field.question == theirs.question_span
        spans = dict(zip(theirs.option_ids, theirs.option_spans))
        assert list(field.options) == [spans[key] for key in q.option_keys]


def test_long_states_are_truncated_like_the_reference(folder, reference):
    _, row, record = ours_and_theirs(folder, reference, state="overdue invoice " * 400)
    from transformers import AutoTokenizer

    from dspy_wsp.decision.families.clef import encoding
    from dspy_wsp.decision.schema import parse_question

    parsed = {name: parse_question(q) for name, q in QUESTIONS.items()}
    short = encoding.encode_request(encoding.Tokenizer(folder), "overdue invoice " * 400, parsed, max_length=500)
    theirs = reference.encode_record(AutoTokenizer.from_pretrained(folder), {"state": "overdue invoice " * 400, "questions": QUESTIONS}, max_length=500)
    assert row.ids == record.input_ids and short.ids == theirs.input_ids and len(short.ids) == 500


def test_head_matches_the_reference(folder, reference):
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import torch
    from safetensors.torch import load_file

    from dspy_wsp.decision.families.clef.mlx import ClefMLX

    parsed, row, record = ours_and_theirs(folder, reference)
    config = json.loads(open(f"{folder}/joint_head_config.json").read())
    weights = {k: v.float() for k, v in load_file(f"{folder}/joint_head.safetensors").items()}
    theirs = reference.JointSchemaHead(**config).eval()
    theirs.load_state_dict(weights, strict=True)

    rng = np.random.default_rng(0)
    hidden = rng.normal(0, 2.0, (len(row.ids), config["hidden_size"])).astype(np.float32)
    table = rng.normal(0, 0.05, (4096, config["hidden_size"])).astype(np.float32)  # a stand-in output embedding
    ids = np.array(row.ids) % len(table)

    with torch.inference_mode():
        expected = theirs(torch.from_numpy(hidden)[None], torch.from_numpy(ids)[None], torch.ones(1, len(ids), dtype=torch.long),
                          [record], torch.from_numpy(table))[0]
    by_name = dict(zip([q.question_id for q in record.questions], zip(record.questions, expected)))

    ours = ClefMLX(**config)
    ours._stored = {k: mx.float32 for k in weights}
    ours.load_named({k: mx.array(v.numpy()) for k, v in weights.items()}, strict=True)
    lexical = [mx.stack([mx.array(table[ids[a:b]].mean(axis=0)) for a, b in f.options]) for f in row.fields]
    recs = ours.score(mx.array(hidden), row, lexical)
    for (name, q), rec in zip(parsed.items(), recs):
        question, logits = by_name[name]
        want = dict(zip(question.option_ids, logits.numpy()))
        np.testing.assert_allclose(np.array(rec["logits"]), [want[key] for key in q.option_keys], atol=2e-3, rtol=2e-3)
        assert rec["head_inputs"].shape == (len(q.option_keys), 4 * config["width"])
