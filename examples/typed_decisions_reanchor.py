"""Compare four versions of a decision model on Typed Decisions: plain, ReAnchor, WeightSpace, and both.

ReAnchor (dspy.experimental.ReAnchor) leaves the weights alone. It fits how each
Predict turns probabilities into answers: a threshold for every noul, cuts for
every score and a weight for every choice option. WeightSpace edits the
weights. This script scores all four combinations on the test split with one
metric, the share of decisions whose decided answer is the gold distribution's
most likely option.

    python examples/typed_decisions_reanchor.py --model nanojev --wsp-run runs/typed-decisions/nanojev

ReAnchor fits on the same practice cases the WeightSpace run used and reports
on the same validation cases; nothing is fitted on the test split. The script
also reports the probability-based accuracy of `typed_decisions.metric`, which
ReAnchor cannot change.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from dspy.experimental import Choice, Noul, ReAnchor, Score
from typed_decisions import TypedDecisions, load_split, make_backend, metric, predicted_probabilities, run_name

from dspy_wsp import stats
from dspy_wsp.patch import Patch


def decided(value) -> str:
    """The answer the program decided, keyed as the dataset keys its gold options."""
    if isinstance(value, Noul):
        return "true" if value.value else "false"
    if isinstance(value, Score):
        return str(value.level)
    if isinstance(value, Choice):
        return str(value.value)
    raise TypeError(f"not a decision output: {type(value).__name__}")


def decided_metric(example, pred, trace=None) -> float:
    """Share of the case's decisions whose decided answer is the gold distribution's most likely option."""
    right = [decided(pred[qid]) == max(g["probabilities"], key=g["probabilities"].get) for qid, g in example.gold.items()]
    return sum(right) / len(right)


def split_from_run(run_dir: Path, train: list[dspy.Example]) -> tuple[list, list]:
    """The practice and validation cases of a WeightSpace run (its practice file lists ids train_<index>)."""
    ids = [json.loads(line)["id"] for line in open(run_dir / "work" / "practice" / "trainset.jsonl")]
    practice_idx = {int(i.removeprefix("train_")) for i in ids}
    return [train[i] for i in sorted(practice_idx)], [ex for i, ex in enumerate(train) if i not in practice_idx]


def evaluate(program, cases: list[dspy.Example]) -> dict:
    def one(ex):
        pred = program(**ex.inputs())
        per_type = {}
        for qid, g in ex.gold.items():
            target = max(g["probabilities"], key=g["probabilities"].get)
            probs = predicted_probabilities(pred[qid])
            per_type.setdefault(g["type"], []).append((decided(pred[qid]) == target, max(probs, key=probs.get) == target))
        return ex.case_id, decided_metric(ex, pred), metric(ex, pred).score, per_type

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(one, cases))
    by_type: dict = {}
    for _, _, _, per_type in rows:
        for kind, pairs in per_type.items():
            by_type.setdefault(kind, []).extend(pairs)
    return {
        "decided": {cid: [d] for cid, d, _, _ in rows},
        "probability": {cid: [p] for cid, _, p, _ in rows},
        "by_type": {k: {"decided_accuracy": sum(a for a, _ in v) / len(v), "probability_accuracy": sum(b for _, b in v) / len(v), "decisions": len(v)}
                    for k, v in sorted(by_type.items())},
    }


def summary(result: dict) -> dict:
    d = [v[0] for v in result["decided"].values()]
    p = [v[0] for v in result["probability"].values()]
    return {"decided_accuracy": stats.mean(d), "decided_se": stats.standard_error(d),
            "probability_accuracy": stats.mean(p), "probability_se": stats.standard_error(p), "by_type": result["by_type"]}


def compare(model: str, framework: str | None, wsp_run: Path, out: Path, train: list, test: list, revision: str | None = None) -> dict:
    """Score plain, ReAnchor, WeightSpace and both on `test`; ReAnchor fits on the WeightSpace run's practice cases."""
    # ReAnchor reruns the program many times on cached answers. DSPy's call history keeps a copy of every
    # request and answer on the client and on each predictor, which grows to tens of GB here.
    dspy.configure(disable_history=True)
    out.mkdir(parents=True, exist_ok=True)
    practice, validation = split_from_run(wsp_run, train)
    backend = make_backend(model, framework, revision)
    edits = {n: t.float().numpy() for n, t in Patch.load(wsp_run / "patch").edited_tensors(backend.reader).items()}
    results, fitted, timings = {}, {}, {}
    try:
        for name, use_edit, reanchor in (("plain", False, False), ("reanchor", False, True), ("wsp", True, False), ("wsp_reanchor", True, True)):
            t = time.time()
            backend.set_weights(edits if use_edit else None)
            program = TypedDecisions(train)
            program.set_lm(backend.lm(cache=True))
            if reanchor:
                optimizer = ReAnchor(metric=decided_metric, num_threads=16, log_dir=out / name)
                program = optimizer.compile(program, trainset=practice, valset=validation)
                fitted[name] = optimizer.report
            results[name] = evaluate(program, test)
            timings[name] = round(time.time() - t, 1)
            print(name, json.dumps({k: v for k, v in summary(results[name]).items() if k != "by_type"}), f"{timings[name]}s", flush=True)
    finally:
        backend.close()

    report = {
        "model": backend.model_id,
        "wsp_run": str(wsp_run),
        "practice_cases": len(practice), "validation_cases": len(validation), "test_cases": len(test),
        "conditions": {k: summary(v) for k, v in results.items()},
        "paired_decided": {
            "reanchor_vs_plain": stats.paired(results["plain"]["decided"], results["reanchor"]["decided"]),
            "wsp_vs_plain": stats.paired(results["plain"]["decided"], results["wsp"]["decided"]),
            "wsp_reanchor_vs_plain": stats.paired(results["plain"]["decided"], results["wsp_reanchor"]["decided"]),
            "wsp_reanchor_vs_wsp": stats.paired(results["wsp"]["decided"], results["wsp_reanchor"]["decided"]),
        },
        "reanchor_reports": {k: {kk: v for kk, v in r.items() if kk != "fitted"} | {"fitted_outputs": len(r.get("fitted", []))} for k, r in fitted.items()},
        "seconds": timings,
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("paired_decided", "reanchor_reports")}, indent=1, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nanojev", help='"d1", "nanojev", a Hugging Face repo id or a local checkpoint folder')
    parser.add_argument("--revision")
    parser.add_argument("--framework", choices=["mlx", "torch"])
    parser.add_argument("--wsp-run", type=Path, required=True, help="a finished WeightSpace run folder (its patch and practice split)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    compare(args.model, args.framework, args.wsp_run, args.out or args.wsp_run.parent / f"reanchor_{run_name(args.model)}", load_split("train"), load_split("test"),
            revision=args.revision)


if __name__ == "__main__":
    main()
