"""How the DSPy engine turns a backend completion into an answer."""

import dspy
import pytest

from dspy_wsp.backends.base import BackendEngine, LMCall


class _Stub:
    model_id = "stub"
    split_thinking = True

    def __init__(self, completion, finish="stop"):
        self.completion, self.finish = completion, finish

    def render_chat(self, msgs, add_generation_prompt=True):
        return "<|im_start|>assistant\n<think>\n"

    def generate(self, prompt, **kw):
        return LMCall(prompt=prompt, completion=self.completion, finish_reason=self.finish, input_tokens=3, output_tokens=5)


def _answer(completion, finish="stop"):
    program = dspy.Predict("question -> answer")
    lm = dspy.LM("wsp/stub", engine=BackendEngine(_Stub(completion, finish)), cache=False)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        return program(question="q").answer


def test_answer_comes_after_think():
    assert _answer("[[ ## answer ## ]]\nwrong\n</think>\n[[ ## answer ## ]]\nright\n[[ ## completed ## ]]") == "right"


def test_cut_while_thinking_gives_no_answer():
    with pytest.raises(dspy.utils.exceptions.AdapterParseError):
        _answer("[[ ## answer ## ]]\nlucky\n[[ ## completed ## ]] still thinking", finish="length")
