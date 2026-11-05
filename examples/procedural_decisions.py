"""Edit a decision model's weights on procedurally generated decision problems, and compare with ReAnchor.

Procedural Typed Decisions (tasksource/procedural-typed-decisions, Apache-2.0)
gives one structured state and several typed questions whose answers follow
exactly from rules the state spells out. Labels are noise-free and balanced:
no answer dominates a question. This script uses the three tasks whose
questions and options are the same in every state, so that ReAnchor can fit
per-question settings:

  event_state_reconstruction       owner (choice), open (noul), severity (score), from an event log
  multi_view_adjudication          intent (choice), urgent (noul), workflow impact (score), from signals and rules
  partial_observation_calibration  incident real (noul, an exact Bayesian posterior), from sensor reports

Practice and validation cases come from the dataset's validation split (400 per
task); the 400 test cases come from its test split. Steps:

    python examples/procedural_decisions.py --model d1 --step baseline-protocol
    python examples/procedural_decisions.py --model d1 --step baseline-program
    OPENAI_API_KEY=... python examples/procedural_decisions.py --model d1 --step optimize
    python examples/procedural_decisions.py --model d1 --step compare
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import dspy
from typed_decisions import baseline, make_backend, make_search, optimize, run_name
from typed_decisions_reanchor import compare

DATASET = "tasksource/procedural-typed-decisions"
TASKS = ("event_state_reconstruction", "multi_view_adjudication", "partial_observation_calibration")
RUNS = Path("runs/procedural-decisions")
POOL_PER_TASK = 400
TEST_CASES = 400
TASK = ("The program answers typed questions about procedurally generated records: event logs, support tickets with "
        "explicit rules, and sensor reports. Every answer follows exactly from rules and facts stated in the record.")


def gold_of(answer: dict) -> dict:
    """A reference answer as a gold distribution keyed like the program's options."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return {"type": "noul", "probabilities": {"false": 1 - p, "true": p}}
    return {"type": answer["type"], "probabilities": {str(k): float(v) for k, v in answer["probabilities"].items()}}


def load(split: str, per_task: dict[str, int], seed: int = 0) -> list[dspy.Example]:
    """`per_task[task]` cases of each task from `split`, keeping each task's usual question set."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = [r for r in pq.read_table(hf_hub_download(DATASET, f"all/{split}-00000-of-00001.parquet", repo_type="dataset")).to_pylist() if r["task"] in TASKS]
    out = []
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        usual, _ = Counter(json.dumps(json.loads(r["questions"]), sort_keys=True) for r in rs).most_common(1)[0]
        rs = sorted((r for r in rs if json.dumps(json.loads(r["questions"]), sort_keys=True) == usual), key=lambda r: r["id"])
        random.Random(f"{seed}:{task}:{split}").shuffle(rs)
        for r in rs[: per_task[task]]:
            answers = json.loads(r["answers"])
            out.append(dspy.Example(case_id=r["id"], workflow=task, level=r["level"], state=json.loads(r["state"]),
                                    questions=json.loads(r["questions"]), gold={q: gold_of(a) for q, a in answers.items()}).with_inputs("workflow", "state"))
    return out


def splits() -> tuple[list[dspy.Example], list[dspy.Example]]:
    train = load("validation", {t: POOL_PER_TASK for t in TASKS})
    sizes = {t: TEST_CASES // len(TASKS) + (i < TEST_CASES % len(TASKS)) for i, t in enumerate(TASKS)}
    return train, load("test", sizes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="d1", help='"d1", "nanojev", a Hugging Face repo id or a local checkpoint folder')
    parser.add_argument("--revision", help="the repo revision (aliases default to the tested one)")
    parser.add_argument("--framework", choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--step", required=True, choices=["baseline-protocol", "baseline-program", "optimize", "compare"])
    parser.add_argument("--passes", type=int, help="the forward-pass budget (default: 25, or what --rounds needs with GEPA)")
    parser.add_argument("--search", choices=["conversation", "gepa"], default="conversation")
    parser.add_argument("--rounds", type=int, default=10, help="with --search gepa: the rounds the budget must pay for")
    args = parser.parse_args()

    train, test = splits()
    name = run_name(args.model)
    search = make_search(args.search, args.rounds) if args.step == "optimize" else None
    run_dir = RUNS / (f"{name}-gepa" if search else name)
    if args.step == "compare":
        compare(args.model, args.framework, run_dir, RUNS / f"reanchor_{name}", train, test, revision=args.revision)
        return
    backend = make_backend(args.model, args.framework, args.revision)
    try:
        if args.step == "optimize":
            passes = args.passes or (search.min_passes() if search else 25)
            optimize(backend, train, test, run_dir=run_dir, passes=passes, holdout=300, task=TASK, search=search)
        else:
            kind = args.step.removeprefix("baseline-")
            baseline(backend, kind, train, test, RUNS / f"baseline_{name}_{kind}.json")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
