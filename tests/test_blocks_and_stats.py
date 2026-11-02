from dspy_wsp.blocks import parse_reply
from dspy_wsp.stats import paired


def test_parse_python_with_explanation():
    r = parse_reply("Look at the norms.\n```python\nx = 1\nx\n```\n")
    assert r.kind == "python" and r.body == "x = 1\nx" and r.explanation == "Look at the norms."


def test_parse_forward():
    r = parse_reply('Measure.\n```forward\n{"type": "generate"}\n```')
    assert r.kind == "forward" and r.body.startswith("{")


def test_parse_final_and_invalid():
    assert parse_reply("I edited layer 3. Gain 0.1 +- 0.02.").kind == "final"
    assert parse_reply("a\n```python\n1\n```\nb\n```python\n2\n```").kind == "invalid"
    assert parse_reply("a\n```python\n1\n").kind == "invalid"
    # other code fences are not actions
    assert parse_reply("Final. The edit was:\n```json\n{}\n```").kind == "final"


def test_paired_counts_and_se():
    o = {"a": [0, 0], "b": [1, 1], "c": [0, 1]}
    e = {"a": [1, 1], "b": [1, 0], "c": [1, 1]}
    p = paired(o, e)
    assert p["items"] == 3
    assert abs(p["gain"] - (1 + -0.5 + 0.5) / 3) < 1e-9
    assert p["improved"] == 3 and p["worsened"] == 1
    assert p["se"] > 0
