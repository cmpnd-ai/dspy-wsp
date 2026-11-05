import math

import pytest

from dspy_wsp.decision.schema import (
    RequestError,
    choice_confidence,
    make_answer,
    parse_question,
    parse_request,
    score_confidence,
)


def test_choice_confidence_values():
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert choice_confidence([0.5, 0.25, 0.25]) == pytest.approx(0.25)


def test_score_confidence_values():
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_make_answer_shapes():
    choice = parse_question({"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B", "c": "C"}})
    a = make_answer(choice, [0.2, 0.7, 0.1])
    assert a["choice"] == "b" and set(a) == {"type", "choice", "probabilities", "confidence"}
    assert list(a["probabilities"]) == ["a", "b", "c"]

    score = parse_question({"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]})
    s = make_answer(score, [0.1, 0.3, 0.6])
    assert set(s["probabilities"]) == {"0", "1", "2"}
    assert s["score"] == pytest.approx(0.3 + 1.2)
    assert set(s) == {"type", "score", "probabilities", "confidence", "legend"} and s["legend"]["2"] == "high"

    noul = parse_question({"type": "noul", "instructions": "Is it?"})
    assert noul.option_keys == ["false", "true"]
    n = make_answer(noul, [0.25, 0.75])
    assert n == {"type": "noul", "noul": pytest.approx(0.75)}


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "choice", "instructions": "x", "criteria": {"only": "one"}},
        {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
        {"type": "pick", "instructions": "x"},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},
        "not a dict",
    ],
)
def test_parse_rejects_bad_questions(bad):
    with pytest.raises(RequestError):
        parse_question(bad)


def test_noul_yes_no_keys_map_to_true_false():
    q = parse_question({"type": "noul", "instructions": "x", "criteria": {"yes": "it holds", "no": "it does not"}})
    assert q.criteria == {"true": "it holds", "false": "it does not"}


def test_parse_request():
    state, qs = parse_request({"model": "m", "state": {"a": 1}, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert state == {"a": 1} and list(qs) == ["q"]
    with pytest.raises(RequestError):
        parse_request({"state": "s", "questions": {}})


def test_make_answer_rejects_bad_probabilities():
    q = parse_question({"type": "noul", "instructions": "x"})
    with pytest.raises(ValueError):
        make_answer(q, [0.5])
    with pytest.raises(ValueError):
        make_answer(q, [math.nan, 1.0])
