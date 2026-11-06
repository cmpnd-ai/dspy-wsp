"""Edit a small decision model's weights to improve the Typed Decisions benchmark.

Typed Decisions (LocalLLaMA/typed-decisions, Apache-2.0) gives a model one state
from one of four workflows and asks five typed questions about it (choice, noul
and score). Every decision has a gold distribution. This example runs the
decision model behind a local System One server, calls it from a DSPy program
through DSPy's TypeSafe client, and lets a writer edit its weights.

    pip install -e ".[decision,examples]"
    python examples/typed_decisions.py --baseline protocol        # the benchmark's own request format
    python examples/typed_decisions.py --baseline program         # through the DSPy program
    OPENAI_API_KEY=... python examples/typed_decisions.py --smoke  # a short end-to-end run
    OPENAI_API_KEY=... python examples/typed_decisions.py          # 25 passes, final check on the test split
    OPENAI_API_KEY=... python examples/typed_decisions.py --search gepa --rounds 10   # GEPA's population search

The writer practices on 900 training cases and validates on the other 300; the
400 test cases are used only in the final check. Accuracy follows the dataset's
scorer (bench/score_with_jobs.py): a decision is right when the most likely
predicted option is the gold distribution's most likely option.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from dspy.experimental import Choice, Noul, Score

from dspy_wsp import GEPASearch, Limits, WeightSpace

DATASET = "LocalLLaMA/typed-decisions"
RUNS = Path("runs/typed-decisions")
WRITER = "openai/gpt-6.1-sol"
VALIDATION = 300
TASK = ("The program answers five typed questions about a business record (an agent trace, a customer message, an invoice "
        "or a security alert). Each question's answer is scored against a gold probability distribution.")


# --- data -----------------------------------------------------------------------------
def load_split(split: str) -> list[dspy.Example]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
    return [
        dspy.Example(case_id=r["id"], workflow=r["workflow"], state=json.loads(r["state"]),
                     questions=json.loads(r["questions"]), gold=json.loads(r["gold"])).with_inputs("workflow", "state")
        for r in pq.read_table(path).to_pylist()
    ]


# --- the program ----------------------------------------------------------------------
def decision_type(q: dict):
    if q["type"] == "choice":
        return Choice[tuple(q["criteria"].items())]
    if q["type"] == "score":
        return Score[tuple(q["criteria"])]
    crit = q.get("criteria") or {}
    pairs = [(value, crit[key]) for key, value in (("true", True), ("false", False)) if crit.get(key)]
    return Noul[tuple(pairs)] if pairs else Noul


def make_signature(workflow: str, questions: dict):
    fields = {"state": (dict, dspy.InputField(desc="The record to judge."))}
    for name, q in questions.items():
        fields[name] = (decision_type(q), dspy.OutputField(desc=q["instructions"]))
    return dspy.make_signature(fields, instructions=f"Answer each question about this {workflow.replace('_', ' ')} record.")


class TypedDecisions(dspy.Module):
    """One Predict per workflow; each case goes to its workflow's questions."""

    def __init__(self, examples: list[dspy.Example]):
        super().__init__()
        schemas = {}
        for ex in examples:
            schemas.setdefault(ex.workflow, ex.questions)
        self.workflows = sorted(schemas)
        for wf in self.workflows:
            setattr(self, wf, dspy.Predict(make_signature(wf, schemas[wf])))

    def forward(self, workflow: str, state: dict):
        return getattr(self, workflow)(state=state)


# --- scoring (the dataset scorer's formulas) ------------------------------------------
def score_one(pred_probs: dict, gold: dict) -> tuple[bool, float, float, float]:
    opts = list(gold["probabilities"])
    g = [float(gold["probabilities"][o]) for o in opts]
    p = [float(pred_probs.get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return p.index(max(p)) == g.index(max(g)), kl, brier, max(p)


def summarize(results: list[tuple[bool, float, float, float]]) -> dict:
    n = len(results)
    bins = [[0, 0, 0.0] for _ in range(10)]
    for ok, _, _, conf in results:
        b = bins[min(int(conf * 10), 9)]
        b[0], b[1], b[2] = b[0] + 1, b[1] + int(ok), b[2] + conf
    ece = sum(t / n * abs(h / t - c / t) for t, h, c in bins if t)
    return {"decisions": n, "accuracy": sum(r[0] for r in results) / n, "kl_from_gold": sum(r[1] for r in results) / n,
            "brier": sum(r[2] for r in results) / n, "ece": ece}


def predicted_probabilities(value) -> dict:
    """A decision output's distribution, keyed as the dataset keys its gold options."""
    if isinstance(value, Noul) or getattr(value, "probability", None) is not None:
        p = float(value.probability)
        return {"true": p, "false": 1 - p}
    return {str(k): float(v) for k, v in (value.probabilities or {}).items()}


def metric(example, pred, trace=None):
    results, notes = [], []
    for qid, gold in example.gold.items():
        probs = predicted_probabilities(pred[qid])
        ok, _, brier, _ = score_one(probs, gold)
        results.append((ok, brier))
        best = max(probs, key=probs.get)
        target = max(gold["probabilities"], key=gold["probabilities"].get)
        notes.append(f"{qid}: {'right' if ok else 'WRONG'} (predicted {best} {probs[best]:.2f}, gold {target} {gold['probabilities'][target]:.2f})")
    accuracy = sum(ok for ok, _ in results) / len(results)
    brier = sum(b for _, b in results) / len(results)
    return dspy.Prediction(score=accuracy, feedback="; ".join(notes) + f". Mean Brier {brier:.3f}.")


# --- baselines --------------------------------------------------------------------------
def protocol_baseline(backend, cases: list[dspy.Example]) -> dict:
    """The dataset scorer's protocol: one request per case, its own state and questions, straight to the server."""
    import requests

    url = backend.lm().base_url + "/v1/systemone"

    def one(ex):
        r = requests.post(url, json={"model": backend.model_id, "state": ex.state, "questions": ex.questions}, timeout=600)
        r.raise_for_status()
        answers = r.json()["answers"]
        return [score_one(answers.get(qid, {}).get("probabilities", {}), gold) for qid, gold in ex.gold.items()]

    with ThreadPoolExecutor(16) as pool:
        return summarize([r for rs in pool.map(one, cases) for r in rs])


def program_baseline(backend, cases: list[dspy.Example], train: list[dspy.Example]) -> dict:
    program = TypedDecisions(train)
    program.set_lm(backend.lm())

    def one(ex):
        pred = program(**ex.inputs())
        return [score_one(predicted_probabilities(pred[qid]), gold) for qid, gold in ex.gold.items()]

    with ThreadPoolExecutor(16) as pool:
        return summarize([r for rs in pool.map(one, cases) for r in rs])


# --- the run ----------------------------------------------------------------------------
def make_backend(model: str, framework: str | None, revision: str | None = None):
    """A backend for an alias ("d1", "nanojev"), a Hugging Face repo id or a local checkpoint folder."""
    from dspy_wsp.decision.loading import load_decision_backend

    return load_decision_backend(model, revision=revision, framework=framework)


def run_name(model: str) -> str:
    """A folder name for a model's runs: the alias, or the repo or folder's last part."""
    return model if "/" not in model else Path(model.rstrip("/")).name.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="d1", help='"d1", "nanojev", a Hugging Face repo id or a local checkpoint folder')
    parser.add_argument("--revision", help="the repo revision (aliases default to the tested one)")
    parser.add_argument("--framework", choices=["mlx", "torch"])
    parser.add_argument("--baseline", choices=["protocol", "program"], help="only score the unedited model on the test split")
    parser.add_argument("--smoke", action="store_true", help="3 passes on a 250-case subset, a 40-case final check")
    parser.add_argument("--passes", type=int, help="the forward-pass budget (default: 25, or what --rounds needs with GEPA)")
    parser.add_argument("--search", choices=["conversation", "gepa"], default="conversation")
    parser.add_argument("--rounds", type=int, default=10, help="with --search gepa: the rounds the budget must pay for")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    train, test = load_split("train"), load_split("test")
    backend = make_backend(args.model, args.framework, args.revision)
    try:
        if args.baseline:
            t = time.time()
            fn = protocol_baseline if args.baseline == "protocol" else lambda b, c: program_baseline(b, c, train)
            result = {"model": backend.model_id, "framework": backend.framework, "baseline": args.baseline, **fn(backend, test),
                      "seconds": round(time.time() - t, 1)}
            out = RUNS / f"baseline_{run_name(args.model)}_{args.baseline}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1))
            print(json.dumps(result, indent=1))
            return
        search = make_search(args.search, args.rounds)
        passes = args.passes or (search.min_passes() if search else 25)
        holdout, valset, limits = VALIDATION, test, Limits(max_generations=900, max_generations_both=450, max_reads=1024)
        if args.smoke:
            by_wf: dict = {}
            for ex in train:
                by_wf.setdefault(ex.workflow, []).append(ex)
            train = [ex for exs in by_wf.values() for ex in exs[:62]] + by_wf[sorted(by_wf)[0]][62:64]
            by_wf = {}
            for ex in test:
                by_wf.setdefault(ex.workflow, []).append(ex)
            holdout, valset = 50, [ex for exs in by_wf.values() for ex in exs[:10]]
            limits = Limits(max_generations=64, max_generations_both=64, max_reads=256)
            search = GEPASearch(min_rounds=2, passes_per_proposal=1) if search else None
            passes = search.min_passes() if search else 3
        suffix = ("-gepa" if search else "") + ("-smoke" if args.smoke else "")
        run_dir = args.run_dir or RUNS / f"{run_name(args.model)}{suffix}"
        optimize(backend, train, valset, run_dir=run_dir, passes=passes, limits=limits, holdout=holdout, task=TASK, search=search)
    finally:
        backend.close()


def make_search(kind: str, rounds: int) -> GEPASearch | None:
    """GEPA's population search with `rounds` guaranteed rounds, or None for one conversation."""
    if kind == "conversation":
        return None
    log = logging.getLogger("dspy_wsp.search")  # one line per round
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    return GEPASearch(min_rounds=rounds)


def optimize(backend, train, valset, *, run_dir, passes=25, limits=None, holdout=VALIDATION, task=TASK, search=None):
    """Run WeightSpace with the writer on a decision benchmark whose examples carry `workflow`, `state`, `questions` and `gold`."""
    optimizer = WeightSpace(
        metric=metric,
        writer_lm=dspy.LM(WRITER, model_type="responses", max_tokens=128_000, timeout=1800),  # Responses API: on Chat Completions gpt-6.1-sol often reasons to the output limit (CMP-1785)
        backend=backend,
        max_passes=passes,
        limits=limits or Limits(max_generations=900, max_generations_both=450, max_reads=1024),
        task_description=task,
        holdout=holdout,
        holdout_key=lambda ex: ex.workflow,
        sandbox="none" if sys.platform == "darwin" else "auto",  # bubblewrap is Linux-only
        run_dir=run_dir,
        resume=True,
        search=search,
    )
    compiled = optimizer.compile(TypedDecisions(train), trainset=train, valset=valset)
    print(compiled.wsp_patch.summary())
    print(json.dumps(compiled.wsp_report["validation"]["paired"], indent=1))
    compiled.save(str(Path(run_dir) / "program.json"))
    if len(compiled.wsp_patch):
        out = compiled.wsp_patch.write_checkpoint(backend.weights_dir, Path(run_dir) / "checkpoint")
        print(f"edited checkpoint: {out}")
    return compiled


def baseline(backend, kind: str, train, test, out: Path) -> dict:
    """Score the unedited model on `test` with the scorer's protocol or through the program, and save the result."""
    t = time.time()
    scores = protocol_baseline(backend, test) if kind == "protocol" else program_baseline(backend, test, train)
    result = {"model": backend.model_id, "framework": backend.framework, "baseline": kind, **scores, "seconds": round(time.time() - t, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
