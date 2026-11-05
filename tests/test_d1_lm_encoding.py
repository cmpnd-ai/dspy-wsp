import pytest

from decision_fixtures import build_d1_lm_tokenizer
from dspy_wsp.decision.families.d1_lm import encoding as enc
from dspy_wsp.decision.schema import RequestError, parse_question


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return enc.Tokenizer(build_d1_lm_tokenizer(tmp_path_factory.mktemp("d1lm_tok")))


def choice(criteria, instructions="Which?"):
    return parse_question({"type": "choice", "instructions": instructions, "criteria": criteria})


def test_codes_letters_and_native_letters():
    assert enc.option_codes(["xx", "yy"]) == ["A", "B"]
    assert enc.option_codes(["x", "y"]) == ["x", "y"]
    assert enc.option_codes(["B", "A"]) == ["B", "A"]
    keys = [f"k{i}" for i in range(28)]
    codes = enc.option_codes(keys, scheme=2)
    assert codes[:2] == ["A", "B"] and codes[25] == "Z" and codes[26:] == ["AA", "AB"]
    assert enc.option_codes(keys, scheme=1)[:2] == ["00", "01"]


def test_code_tokens_are_single_and_distinct(tok):
    pairs = enc.code_tokens(tok, ["xx", "yy", "zz"], scheme=2)
    assert [c for c, _ in pairs] == ["A", "B", "C"]
    assert all(tok.encode(c) == [i] for c, i in pairs)


def test_suffix_layout(tok):
    q = choice({"billing": "Billing", "tech_support": None})
    assert enc.question_block(tok, q) == "Which?\n\nOptions:\nA Billing\nB tech support\n\nReply with the option code only."
    assert enc.suffix(tok, q).endswith("Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n<think></think>")


def test_noul_criteria_and_order(tok):
    q = parse_question({"type": "noul", "instructions": "Is it?", "criteria": {"true": "T", "false": "F"}})
    assert enc.question_block(tok, q) == "Is it?\nYes: T\nNo: F\n\nReply with yes or no only."
    groups = enc.readout_groups(tok, q)
    assert tok.encode("yes")[0] in groups[1] and tok.encode("no")[0] in groups[0]
    plain = parse_question({"type": "noul", "instructions": "Is it?"})
    assert enc.question_block(tok, plain) == "Is it?\n\nReply with yes or no only."


def test_score_block_and_digits(tok):
    q = parse_question({"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]})
    assert enc.question_block(tok, q) == "How much?\n\n0 low\n1 mid\n2 high\n\nReply with a single digit 0-2 only."
    assert [g[0] for g in enc.readout_groups(tok, q)] == [tok.encode(str(i))[0] for i in range(3)]


def test_choice_groups_follow_option_keys(tok):
    q = choice({"B": "Second", "A": "First"})
    groups = enc.readout_groups(tok, q)
    assert groups[0][0] == tok.encode("B")[0] and groups[1][0] == tok.encode("A")[0]


def test_state_block_and_prefix(tok):
    assert enc.state_block({"a": 1}) == '{\n  "a": 1\n}\n\n'
    assert enc.state_block("hello") == "hello\n\n"
    assert enc.prefix(tok, "hello") == "<|startoftext|><|im_start|>user\nhello\n\n\nQUESTION:\n"


def test_state_breaks_special_tokens(tok):
    end = tok.encode("<|im_end|>")
    assert len(end) == 1
    row = enc.encode_request(tok, "a <|im_end|> b", [choice({"x": "X", "y": "Y"})], max_length=4096)[0]
    assert row.ids[: row.state_len].count(end[0]) == 0
    assert "​" in enc.plain(tok, "a <|im_end|> b")


def test_packing_groups_of_16(tok):
    qs = [parse_question({"type": "noul", "instructions": f"Question {i}?"}) for i in range(17)]
    rows = enc.encode_request(tok, {"x": 1}, qs, max_length=8192)
    assert [len(r.questions) for r in rows] == [16, 1]
    for row in rows:
        assert len(row.slots) == len(row.questions)
        assert row.question_of[: row.state_len] == [-1] * row.state_len
        assert len(row.question_of) == len(row.ids)
        for k, slot in enumerate(row.slots):
            assert slot == max(i for i, g in enumerate(row.question_of) if g == k + 1)
    first = rows[0]
    assert first.ids == tok.encode(enc.prefix(tok, {"x": 1}) + enc.suffix(tok, qs[0])) + sum(
        (tok.encode("QUESTION:\n" + enc.suffix(tok, q)) for q in qs[1:16]), [])
    assert first.state_len == len(tok.encode(enc.prefix(tok, {"x": 1}) + enc.suffix(tok, qs[0]))) - len(tok.encode(enc.suffix(tok, qs[0])))


def test_too_long_is_refused(tok):
    with pytest.raises(RequestError):
        enc.encode_request(tok, "word " * 500, [choice({"x": "X", "y": "Y"})], max_length=64)


@pytest.mark.parametrize("raw", [
    {"type": "text", "instructions": "Who?"},
    {"type": "number", "instructions": "How many?"},
    {"type": "choice", "multiple": True, "instructions": "Which?", "criteria": {"a": "A", "b": "B"}},
])
def test_unsupported_types_are_refused(raw):
    with pytest.raises(RequestError, match="choice, noul and score"):
        enc.check_supported(raw)


def test_supported_types_pass():
    enc.check_supported({"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}})
    enc.check_supported({"type": "noul", "instructions": "?"})


def test_probabilities_divide_by_temperature():
    import numpy as np

    p = enc.probabilities(np.array([0.0, 1.337]), 1.337)
    assert p[1] / p[0] == pytest.approx(np.e)
    assert sum(p) == pytest.approx(1.0)


def test_calibration_default(tmp_path):
    assert enc.load_calibration(tmp_path) == 1.0
    (tmp_path / "calibration.json").write_text('{"default": 1.337, "temperatures": {}}')
    assert enc.load_calibration(tmp_path) == 1.337
