"""Open any supported decision model by name, repo or folder, and check that it works.

    backend = load_decision_backend("nanojev")                 # the pinned NanoJev checkpoint
    backend = load_decision_backend("someone/d1-finetune")     # family read from the checkpoint's config
    report = check_decision_model("/path/to/checkpoint")       # load, answer, batch and edit checks
    print(report.summary())

The families, how each is recognized and their aliases are listed in
`dspy_wsp.decision.families`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dspy_wsp.decision.families import FAMILIES, backend_class

ALIASES = {alias: target for f in FAMILIES for alias, target in f.aliases.items()}
FAMILY_HINT = "; ".join(f"{f.name} ({f.hint})" for f in FAMILIES)


class DecisionModelError(ValueError):
    """The model cannot be opened as a supported decision model."""


def resolve_model(model: str, revision: str | None) -> tuple[str, str | None]:
    """(repo or folder, revision). Aliases and the tested repos default to their pinned revisions."""
    if model in ALIASES:
        repo, pinned = ALIASES[model]
        return repo, revision or pinned
    for repo, pinned in ALIASES.values():
        if model == repo:
            return repo, revision or pinned
    return model, revision


def _read_json(model: str, revision: str | None, name: str) -> dict | None:
    path = Path(model)
    if path.is_dir():
        f = path / name
        return json.loads(f.read_text()) if f.is_file() else None
    if path.exists() or model.count("/") != 1:
        return None
    from huggingface_hub import hf_hub_download

    try:
        return json.loads(Path(hf_hub_download(model, name, revision=revision)).read_text())
    except Exception:  # noqa: BLE001 - a missing file or repo means "not this family"
        return None


def detect_family(model: str, revision: str | None = None) -> str:
    """The name of the family whose config `model` matches."""
    if _read_json(model, revision, "config.json") is None:
        raise DecisionModelError(f"{model}: no config.json found (expected a local folder or a Hugging Face repo id)")
    for family in FAMILIES:
        if family.matches(lambda name: _read_json(model, revision, name)):
            return family.name
    raise DecisionModelError(f"{model} is not a supported decision model. Supported families: {FAMILY_HINT}.")


def load_decision_backend(model: str, *, revision: str | None = None, framework: str | None = None,
                          device: str | None = None, max_length: int | None = None):
    """A decision backend for `model`: an alias in `ALIASES` ("d1", "nanojev", "d1-2b"), a Hugging Face repo id or a
    local folder."""
    repo, revision = resolve_model(model, revision)
    name = detect_family(repo, revision)
    cls = backend_class(next(f for f in FAMILIES if f.name == name))
    kwargs = {"framework": framework, "device": device}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return cls(repo, revision=revision, **kwargs)


# --- compatibility check ------------------------------------------------------------------

SAMPLE_STATE = {"ticket": "Customer reports that invoice 4471 charged them twice for the March plan.",
                "account": {"plan": "team", "seats": 12}}
SAMPLE_QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Billing", "technical": "Technical support", "sales": "Sales"}},
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Not urgent: it can wait a week", "Somewhat urgent: handle within a day", "Urgent: handle now"]},
}
SECOND_STATE = {"ticket": "The export button returns a 500 error since this morning's release.", "account": {"plan": "free", "seats": 1}}


@dataclass
class CompatibilityReport:
    model: str
    revision: str | None = None
    family: str | None = None
    framework: str | None = None
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        return ok

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    @property
    def problems(self) -> list[str]:
        return [f"{c['name']}: {c['detail']}" for c in self.checks if not c["ok"]]

    def summary(self) -> str:
        head = f"{self.model} ({self.family or 'unknown family'}, {self.framework or '-'}): {'PASS' if self.ok else 'FAIL'}"
        return "\n".join([head] + [f"  {c['name']:<9} {'ok  ' if c['ok'] else 'FAIL'} {c['detail']}" for c in self.checks])


def _answer_problems(answer: dict, q: dict) -> list[str]:
    keys = list(q["criteria"]) if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])
    probs = answer.get("probabilities", {})
    out = []
    if [str(k) for k in probs] != keys:
        out.append(f"options {list(probs)} differ from {keys}")
    values = [float(v) for v in probs.values()]
    if not all(math.isfinite(v) and v >= 0 for v in values) or abs(sum(values) - 1) > 1e-4:
        out.append(f"probabilities {values} are not a distribution")
    conf = answer.get("confidence")
    if conf is not None and not 0 <= conf <= 1:
        out.append(f"confidence {conf} is outside [0, 1]")
    return out


def _probs(response: dict) -> np.ndarray:
    return np.array([float(v) for a in response["answers"].values() for v in a["probabilities"].values()])


def _edit_target(backend) -> str | None:
    """The smallest editable matrix outside the protected tensors: cheap to swap, and on the forward path."""
    import re

    protected = [re.compile(p) for p in backend.default_protected()]
    names = [n for n in backend.editable_names() if not any(p.search(n) for p in protected) and len(backend.reader.shape(n)) >= 2]
    return min(names, key=lambda n: (int(np.prod(backend.reader.shape(n))), n)) if names else None


def check_decision_model(model: str, *, revision: str | None = None, framework: str | None = None,
                         device: str | None = None, max_length: int | None = None) -> CompatibilityReport:
    """Load `model` and check that it answers valid typed questions, batches consistently and takes weight edits."""
    repo, revision = resolve_model(model, revision)
    report = CompatibilityReport(repo, revision)
    try:
        report.family = detect_family(repo, revision)
    except DecisionModelError as e:
        report.add("family", False, str(e))
        return report
    try:
        backend = load_decision_backend(repo, revision=revision, framework=framework, device=device, max_length=max_length)
    except Exception as e:  # noqa: BLE001 - any load failure is the finding
        report.add("load", False, f"{type(e).__name__}: {e}")
        return report
    report.framework = backend.framework
    try:
        _run_checks(backend, report)
    finally:
        backend.close()
    return report


def _run_checks(backend, report: CompatibilityReport) -> None:
    d = backend.dims()
    report.add("load", True, f"{d['layers']} layers, width {d['hidden']}, vocab {d['vocab']}")

    mapped = backend.editable_names()
    skipped = [n for n in backend.reader.names if n not in set(mapped)]
    size = sum(int(np.prod(backend.reader.shape(n))) for n in mapped)
    detail = f"{len(mapped)} tensors ({size:,} weights) map to the network"
    report.add("tensors", bool(mapped), detail + (f"; {len(skipped)} unused (e.g. {skipped[0]})" if skipped else ""))

    try:
        first = backend.system_one(SAMPLE_STATE, SAMPLE_QUESTIONS)
    except Exception as e:  # noqa: BLE001
        report.add("answers", False, f"{type(e).__name__}: {e}")
        return
    problems = [f"{name}: {p}" for name, q in SAMPLE_QUESTIONS.items() for p in _answer_problems(first["answers"].get(name, {}), q)]
    report.add("answers", not problems, "; ".join(problems) or "choice, noul and score answers are valid distributions")

    batch = backend.system_one_batch([(SECOND_STATE, SAMPLE_QUESTIONS), (SAMPLE_STATE, SAMPLE_QUESTIONS)])
    gap = float(np.abs(_probs(batch[1]) - _probs(first)).max())
    report.add("batching", gap < 1e-3, f"a request answered alone and in a batch differs by at most {gap:.1e}")

    target = _edit_target(backend)
    if target is None:
        report.add("edits", False, "no editable matrix outside the protected tensors")
        return
    w = backend.reader.get_f32(target)
    rng = np.random.default_rng(0)
    edit = {target: -w + rng.normal(0, float(w.std()) + 0.1, w.shape).astype(np.float32)}
    with backend.using(edit):
        edited = backend.system_one(SAMPLE_STATE, SAMPLE_QUESTIONS)
    restored = backend.system_one(SAMPLE_STATE, SAMPLE_QUESTIONS)
    moved = float(np.abs(_probs(edited) - _probs(first)).max())
    back = float(np.abs(_probs(restored) - _probs(first)).max())
    ok = moved > 1e-4 and back < 1e-5
    report.add("edits", ok, f"editing {target} moved answers by {moved:.1e}; restoring it left {back:.1e}")


def report_dict(report: CompatibilityReport) -> dict[str, Any]:
    return {"model": report.model, "revision": report.revision, "family": report.family, "framework": report.framework,
            "ok": report.ok, "checks": report.checks}
