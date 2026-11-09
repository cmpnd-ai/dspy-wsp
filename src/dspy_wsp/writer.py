"""One chat call to the writer LM."""

from __future__ import annotations


def ask_writer(writer_lm, messages: list[dict]) -> str:
    """Send OpenAI-style chat messages to the writer and return its reply text.

    A `dspy.LM` receives a canonical request built with the LM's own generation
    settings. Any other callable receives `messages=` and returns a list of completions.
    """
    import dspy

    if not isinstance(writer_lm, dspy.BaseLM):
        out = writer_lm(messages=messages)
        first = out[0] if out else ""
        return str(first.get("text") or "") if isinstance(first, dict) else str(first)

    from dspy.adapters import Prompt
    from dspy.clients.requests import build_request
    from dspy.lm15 import Message

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    turns = [Message.assistant(m["content"]) if m["role"] == "assistant" else Message.user(m["content"]) for m in messages if m["role"] != "system"]
    return writer_lm(build_request(writer_lm, Prompt(messages=turns, system=system))).text or ""
