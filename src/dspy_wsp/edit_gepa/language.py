"""The `edits` language from the paper's Experiment 5: assignments to slices of named tensors, written as text.

Programs are parsed with `ast` and never executed. A statement the parser can't read is skipped and reported, as in
the paper. Each statement assigns, multiplies, adds or subtracts a number or a (nested) list of numbers:

    W1[0:60:6, 0] = 6
    b1[0:9] = -6
    W3[29, :] *= 0.5
    W2[3, 4:7] = [1, -2, 0.5]

A block may start with `# base: zero` (or `# base: <version>`), which callers read with `base_of`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping

import numpy as np

BLOCK = re.compile(r"```edits[^\n]*\n(.*?)```", re.S)
BASE = re.compile(r"^\s*#\s*base\s*:\s*(\S+)", re.M)
MAX_ABS = 1e6


def last_block(reply: str | None) -> str | None:
    """The last ```edits block in a reply, or None."""
    blocks = BLOCK.findall(reply or "")
    return blocks[-1] if blocks else None


def base_of(block: str, default: str) -> str:
    m = BASE.search(block)
    return m.group(1).strip() if m else default


def statements(text: str) -> list[str]:
    """Split into statements; a statement continues over lines while brackets are open. Comments are dropped."""
    out, buf, depth = [], [], 0
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if not code.strip() and depth == 0:
            continue
        buf.append(code)
        depth += code.count("[") - code.count("]")
        if depth <= 0:
            out.append(" ".join(buf).strip())
            buf, depth = [], 0
    if buf:
        out.append(" ".join(buf).strip())
    return [s for s in out if s]


def _num(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _num(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("not a number")


def _index(node):
    if isinstance(node, ast.Slice):
        f = lambda x: None if x is None else int(_num(x))  # noqa: E731
        return slice(f(node.lower), f(node.upper), f(node.step))
    v = _num(node)
    if not float(v).is_integer():
        raise ValueError("index must be an integer")
    return int(v)


def _value(node):
    if isinstance(node, ast.List):
        return [_value(e) for e in node.elts]
    v = float(_num(node))
    if abs(v) > MAX_ABS:
        raise ValueError(f"|value| > {MAX_ABS:g}")
    return v


def zeros(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    return {k: np.zeros(s, np.float32) for k, s in shapes.items()}


def apply(text: str, shapes: Mapping[str, tuple[int, ...]], base: Mapping[str, np.ndarray] | None = None
          ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Apply `text` in order to a copy of `base` (all zeros if None). Returns the weights and the skipped statements."""
    net = {k: np.array(v, np.float32) for k, v in (base or zeros(shapes)).items()}
    errors = []
    for i, s in enumerate(statements(text), 1):
        try:
            tree = ast.parse(s, mode="exec").body
            if len(tree) != 1 or not isinstance(tree[0], (ast.Assign, ast.AugAssign)):
                raise ValueError("not an assignment")
            st = tree[0]
            target = st.targets[0] if isinstance(st, ast.Assign) else st.target
            if (isinstance(st, ast.Assign) and len(st.targets) != 1) or not isinstance(target, ast.Subscript) \
                    or not isinstance(target.value, ast.Name) or target.value.id not in net:
                raise ValueError(f"target must be one of {', '.join(net)} with an index")
            sl = target.slice
            idx = tuple(_index(e) for e in sl.elts) if isinstance(sl, ast.Tuple) else (_index(sl),)
            arr = net[target.value.id]
            if len(idx) != arr.ndim:
                raise ValueError(f"{target.value.id} needs {arr.ndim} indices")
            view_shape = arr[idx].shape
            val = np.asarray(_value(st.value), dtype=np.float32)
            if val.ndim and val.shape != view_shape:
                raise ValueError(f"{val.shape[0] if val.ndim == 1 else val.shape} values for {view_shape or 1} entries")
            if isinstance(st, ast.Assign):
                arr[idx] = val
            elif isinstance(st.op, ast.Mult):
                arr[idx] *= val
            elif isinstance(st.op, ast.Add):
                arr[idx] += val
            elif isinstance(st.op, ast.Sub):
                arr[idx] -= val
            else:
                raise ValueError("only =, *=, += and -=")
        except Exception as e:
            errors.append(f"statement {i} skipped ({s[:80]}): {e}")
    return net, errors
