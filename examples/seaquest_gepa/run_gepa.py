"""EditGEPA on MinAtar Seaquest, starting from the paper's hand-written Exp 5 network.

    python run_gepa.py --dry-run                 # free plumbing test: the writer is a stub
    python run_gepa.py --seed 0                  # 6,400 episodes, gpt-6-astra at xhigh
"""
import argparse
import json
import os
import random
import time

import dotenv

import seaquest as S
from dspy_wsp.edit_gepa import EditGEPA

PRICES = {"gpt-6-astra": (10.0, 1.0, 50.0)}  # $/M tokens (input, cached input, output), Azure, 2026-11-05


class StubLM:
    """Stands in for the writer in --dry-run: one small random edit per call."""
    history: list = []

    def __call__(self, messages):
        return [f"Raise the fire preference.\n```edits\nW3[0:8, 5] = {round(random.uniform(25, 35), 1)}\n```"]


def writer(model, effort):
    import dspy

    azure = {"api_base": os.environ["AZURE_API_BASE"], "api_key": os.environ["AZURE_API_KEY"]} \
        if "AZURE_API_BASE" in os.environ else {}
    return dspy.LM(model, reasoning_effort=effort, max_tokens=100_000, cache=False, num_retries=6, timeout=3600, **azure)


if __name__ == "__main__":
    dotenv.load_dotenv(S.HERE / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=6400, help="episodes, including validation")
    ap.add_argument("--lm", default="openai/gpt-6-astra")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        a.budget = min(a.budget, 700)
    run_dir = a.run_dir or str(S.HERE / "runs" / f"gepa_s{a.seed}{'_dry' if a.dry_run else ''}")

    search = EditGEPA(
        writer_lm=StubLM() if a.dry_run else writer(a.lm, a.effort),
        shapes=S.SHAPES, evaluate=S.evaluate, task=S.task(), context=S.context(), protocol=S.PROTOCOL,
        budget=a.budget, units_per_instance=S.GROUP, minibatch_size=8,
        prices=PRICES.get(a.lm.split("/")[-1]), seed=a.seed, run_dir=run_dir)
    res = search.run(S.seed_program(), trainset=S.train_groups(a.seed), valset=S.VAL)

    t = time.time()
    test = S.test(res.best_weights) | {"test_seconds": round(time.time() - t, 1)}
    out = {"seed": a.seed, "best_val": res.best_val, "candidates": len(res.candidates), "episodes_used": res.units_used,
           "writer_calls": res.writer_calls, "writer_tokens": res.writer_tokens, "writer_usd": res.writer_usd,
           "compile_seconds": round(res.seconds), **test}
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
