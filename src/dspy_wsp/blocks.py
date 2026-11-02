"""Parse the writer's replies.

Each turn the writer replies with an explanation and exactly one block:
```python (run in the kernel) or ```forward (a JSON forward-pass request).
A reply with no block is a final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLOCK = re.compile(r"```(python|forward)[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN = re.compile(r"```(python|forward)\b")


@dataclass
class Reply:
    kind: str  # "python", "forward", "final" or "invalid"
    body: str = ""
    explanation: str = ""
    error: str = ""


def parse_reply(text: str) -> Reply:
    blocks = list(_BLOCK.finditer(text))
    if len(blocks) > 1:
        return Reply("invalid", error=f"Reply with exactly one ```python or ```forward block per turn; this reply had {len(blocks)}. Nothing was run.")
    if len(blocks) == 1:
        m = blocks[0]
        return Reply(m.group(1), body=m.group(2).strip("\n"), explanation=text[: m.start()].strip())
    if _OPEN.search(text):
        return Reply("invalid", error="The code block was not closed with ```. Nothing was run.")
    return Reply("final", body=text.strip())
