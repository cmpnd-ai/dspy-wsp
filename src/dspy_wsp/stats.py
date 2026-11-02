"""Scores and paired comparisons for generation passes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def standard_error(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def paired(original: Mapping[str, Sequence[float]], edited: Mapping[str, Sequence[float]]) -> dict:
    """Compare two models on the same items.

    Each item's samples are averaged first, so the standard error is over items
    (samples of one item share the item's difficulty and are not independent).
    `improved` and `worsened` count sample pairs whose score went up or down;
    with a 0/1 metric these are the paper's rescues and losses.
    """
    items = sorted(set(original) & set(edited))
    diffs, o_means, e_means = [], [], []
    improved = worsened = 0
    for item in items:
        o, e = list(original[item]), list(edited[item])
        o_means.append(mean(o))
        e_means.append(mean(e))
        diffs.append(e_means[-1] - o_means[-1])
        for a, b in zip(o, e):
            improved += b > a
            worsened += b < a
    return {
        "items": len(items),
        "original": mean(o_means),
        "edited": mean(e_means),
        "gain": mean(diffs),
        "se": standard_error(diffs),
        "improved": improved,
        "worsened": worsened,
    }
