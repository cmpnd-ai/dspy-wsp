"""GEPA's population search over WeightSpace edits.

The search keeps a population of candidate edits. Each round, GEPA picks a parent
from its Pareto frontier over validation examples and scores it on a minibatch of
practice examples. The writer then gets a short session in the Python kernel that
starts from the parent's edit and notes, and the edit it leaves becomes the child.
GEPA keeps the child only if it beats the parent on the same minibatch, and then
scores it on the validation set. Every scoring and every writer pass counts one
forward pass against the run's budget.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dspy_wsp.budget import Budget
from dspy_wsp.edits import EditError, EditStore
from dspy_wsp.session import Session

logger = logging.getLogger(__name__)


@dataclass
class GEPASearch:
    """Settings for `WeightSpace(search=GEPASearch(...))`.

    Args:
        min_rounds: rounds the budget must pay for; WeightSpace refuses a smaller max_passes.
        minibatch_size: practice examples per parent and child comparison.
        passes_per_proposal: forward passes the writer may spend in one round.
        code_runs_per_proposal: Python runs the writer may spend in one round.
    """

    min_rounds: int = 10
    minibatch_size: int = 8
    passes_per_proposal: int = 2
    code_runs_per_proposal: int = 30

    @property
    def round_cost(self) -> int:
        """The most passes one round can use: parent minibatch, child minibatch, child validation and the writer's passes."""
        return 3 + self.passes_per_proposal

    def min_passes(self) -> int:
        """The seed's validation run plus `min_rounds` rounds at their most expensive."""
        return 1 + self.min_rounds * self.round_cost


def restore_candidate(cand_dir: Path, work_dir: Path) -> None:
    """Make the work folder's edits and notes exactly the candidate's."""
    edited = Path(work_dir) / "edited"
    if edited.exists():
        shutil.rmtree(edited)
    shutil.copytree(Path(cand_dir) / "edited", edited)
    notes, saved = Path(work_dir) / "notes.md", Path(cand_dir) / "notes.md"
    if saved.exists():
        shutil.copyfile(saved, notes)
    elif notes.exists():
        notes.unlink()


def _save_candidate(work_dir: Path, cand_dir: Path, answer: str = "") -> None:
    cand_dir.mkdir(parents=True)
    shutil.copytree(Path(work_dir) / "edited", cand_dir / "edited")
    notes = Path(work_dir) / "notes.md"
    if notes.exists():
        shutil.copyfile(notes, cand_dir / "notes.md")
    (cand_dir / "answer.md").write_text(answer)


class _Search:
    """GEPA's adapter, proposer, stop rule and logging for one WeightSpace run."""

    def __init__(self, settings: GEPASearch, *, writer_lm, kernel, forward, budget: Budget, run_dir: Path, work_dir: Path,
                 round_prompt: str, edit_store: EditStore, handoff_every: int, require_explanations: bool, max_turns: int):
        self.s = settings
        self.writer_lm, self.kernel, self.forward, self.budget = writer_lm, kernel, forward, budget
        self.run_dir, self.work_dir = Path(run_dir), Path(work_dir)
        self.round_prompt, self.edit_store = round_prompt, edit_store
        self.handoff_every, self.require_explanations, self.max_turns = handoff_every, require_explanations, max_turns
        self.cands = self.run_dir / "candidates"
        self.examples = {**forward.practice, **forward.validation}
        self.program = forward._program(None)
        self.round = 0
        self.n_candidates = 0
        self.parent_of: dict[str, str] = {}
        self.val_scores: dict[str, float] = {}
        self.row: dict = {}
        self.rows: list[dict] = []
        self.history_start = len(getattr(writer_lm, "history", None) or [])
        self.error: Exception | None = None  # set when the search must stop: GEPA swallows exceptions and retries
        self.t0 = time.time()

    # --- candidates -------------------------------------------------------------------
    def new_candidate(self, answer: str = "") -> str:
        cid = f"c{self.n_candidates:03d}"
        self.n_candidates += 1
        _save_candidate(self.work_dir, self.cands / cid, answer)
        return cid

    def edits_of(self, cid: str) -> dict:
        return EditStore(self.cands / cid / "edited", self.edit_store.reader, self.edit_store.protected).load_all()

    # --- GEPA's adapter -------------------------------------------------------------------
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        cid = candidate["edit"]
        items = [(i, self.examples[i]) for i in batch]
        try:
            with self.forward.backend.using(self.edits_of(cid)):
                rows = self.forward.run_items(self.program, items, 1, cid)
        except Exception as e:
            self.error = self.error or e
            raise
        finally:
            self.budget.passes_used += 1
        if rows and all("error" in r for r in rows):
            self.error = self.error or RuntimeError(f"every example failed when scoring {cid}, e.g. {rows[0]['error']}")
        scores = [float(r.get("score", 0.0)) for r in rows]
        mean = sum(scores) / max(1, len(scores))
        on_validation = bool(batch) and batch[0] in self.forward.validation
        if on_validation:
            self.val_scores[cid] = mean
            if self.row.get("child") == cid:
                self.row["child_val"] = mean
        elif self.row:
            self.row["parent_minibatch" if cid == self.row["parent"] else "child_minibatch"] = mean
        traces = [self._trace(r) for r in rows] if capture_traces else None
        return EvaluationBatch(outputs=scores, scores=scores, trajectories=traces)

    @staticmethod
    def _trace(row: dict) -> str:
        line = f"{row['id']}: score {row.get('score', 0.0):g}"
        if row.get("feedback"):
            return f"{line}. {row['feedback']}"
        if row.get("error"):
            return f"{line}. The program failed: {row['error']}"
        return f"{line}. Outputs: {json.dumps(row.get('outputs', {}))[:1500]}"

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {"edit": [{"feedback": "\n".join(eval_batch.trajectories or [])}]}

    # --- the proposer ------------------------------------------------------------------
    def propose(self, candidate, reflective_dataset, components_to_update):
        if self.error is not None:  # GEPA retries a failed proposal at once; don't start another session
            raise self.error
        parent = candidate["edit"]
        self.round += 1
        n = self.round
        feedback = reflective_dataset["edit"][0]["feedback"]
        restore_candidate(self.cands / parent, self.work_dir)
        notes_path = self.work_dir / "notes.md"
        notes = notes_path.read_text()[:20_000] if notes_path.exists() else ""
        b = Budget(max_passes=min(self.s.passes_per_proposal, self.budget.passes_left),
                   max_code_runs=min(self.s.code_runs_per_proposal, self.budget.code_runs_left))
        opening = (
            f"{self.round_prompt}\n\nROUND {n}. You start from candidate {parent}.\n\n"
            f"ITS EDIT:\n{self.edit_store.describe()}\n\nITS NOTEBOOK (notes.md):\n{notes or '(empty)'}\n\n"
            f"HOW IT DID ON {feedback.count(chr(10)) + 1} PRACTICE EXAMPLES CHOSEN BY THE SEARCH:\n{feedback}\n\n{b.status()}"
        )
        session = Session(
            writer_lm=self.writer_lm, kernel=self.kernel, forward_tool=self.forward, budget=b,
            run_dir=self.run_dir / "rounds" / f"r{n:02d}", work_dir=self.work_dir, task_prompt=opening,
            edit_store=self.edit_store, handoff_every=self.handoff_every, require_explanations=self.require_explanations,
            max_turns=self.max_turns, max_nudges=0, pass_offset=self.budget.passes_used,
        )
        failure = None
        try:
            answer = session.run()
        except Exception as e:  # e.g. the writer's API failed after its retries
            failure = e
        self.budget.passes_used += b.passes_used
        self.budget.code_runs_used += b.code_runs_used
        if failure is not None:
            self.error = failure
            self.row["error"] = f"{type(failure).__name__}: {failure}"
            self._write_row(accepted=False)
            raise failure
        note = ""
        try:
            self.edit_store.load_all()
        except EditError as e:  # an edit written around save_edit: the child keeps the parent's edit
            note = f"the round's edit was refused ({e}); the child keeps the parent's edit"
            logger.warning("round %d: %s", n, note)
            restore_candidate(self.cands / parent, self.work_dir)
        child = self.new_candidate(answer)
        self.parent_of[child] = parent
        self.row.update(child=child, writer_passes=b.passes_used, writer_code_runs=b.code_runs_used, **({"note": note} if note else {}))
        return {"edit": child}

    # --- stop rule and logging ------------------------------------------------------------
    def should_stop(self, gepa_state=None) -> bool:
        return self.error is not None or self.budget.passes_left < self.s.round_cost or self.budget.code_runs_left == 0

    def on_candidate_selected(self, event) -> None:
        self.row = {"round": self.round + 1, "parent": event["candidate"]["edit"], "parent_minibatch": None,
                    "child": None, "child_minibatch": None, "child_val": None}

    def on_iteration_end(self, event) -> None:
        if self.row.get("child"):
            self._write_row(accepted=bool(event["proposal_accepted"]))
        self.row = {}

    def _write_row(self, accepted: bool) -> None:
        writer = self._writer_usage()
        best = max(self.val_scores.items(), key=lambda kv: kv[1]) if self.val_scores else (None, None)
        row = {**self.row, "accepted": accepted, "best": best[0], "best_val": best[1],
               "passes_used": self.budget.passes_used, "code_runs_used": self.budget.code_runs_used,
               **writer, "seconds": round(time.time() - self.t0, 1)}
        self.rows.append(row)
        with open(self.run_dir / "rounds.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self._write_progress()
        val = "-" if row["child_val"] is None else f"{row['child_val']:.4f}"
        logger.info("round %d: %s -> %s, minibatch %s -> %s, %s, validation %s, best %s (%.4f), passes %d of %d",
                    row["round"], row["parent"], row["child"], row["parent_minibatch"], row["child_minibatch"],
                    "accepted" if row["accepted"] else "rejected", val, row["best"], row["best_val"] or 0,
                    row["passes_used"], self.budget.max_passes)

    def _writer_usage(self) -> dict:
        entries = (getattr(self.writer_lm, "history", None) or [])[self.history_start:]
        tokens_in = sum((e.get("usage") or {}).get("prompt_tokens", 0) or 0 for e in entries)
        tokens_out = sum((e.get("usage") or {}).get("completion_tokens", 0) or 0 for e in entries)
        cost = sum(e.get("cost") or 0.0 for e in entries)
        return {"writer_calls": len(entries), "writer_tokens_in": tokens_in, "writer_tokens_out": tokens_out, "writer_usd": round(cost, 4)}

    def _write_progress(self) -> None:
        fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
        lines = ["# GEPA search progress", "",
                 "| Round | Parent | Child | Parent minibatch | Child minibatch | Accepted | Child validation | Best | Best validation | Passes | Writer $ | Minutes |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['round']} | {r['parent']} | {r['child']} | {fmt(r['parent_minibatch'])} | {fmt(r['child_minibatch'])} | "
                         f"{'yes' if r['accepted'] else 'no'} | {fmt(r['child_val'])} | {r['best']} | {fmt(r['best_val'])} | "
                         f"{r['passes_used']} of {self.budget.max_passes} | {r['writer_usd']:.2f} | {r['seconds'] / 60:.0f} |")
        (self.run_dir / "progress.md").write_text("\n".join(lines) + "\n")


def run_gepa(settings: GEPASearch, *, writer_lm, kernel, forward, budget: Budget, run_dir: str | Path, work_dir: str | Path,
             round_prompt: str, edit_store: EditStore, seed: int = 0, handoff_every: int = 40,
             require_explanations: bool = True, max_turns: int = 1000) -> tuple[dict, str, dict]:
    """Run the search. Returns the best candidate's edits, its writer's final message, and a report."""
    from gepa.api import optimize

    if not forward.validation:
        raise ValueError("GEPA search needs a validation set (holdout)")
    search = _Search(settings, writer_lm=writer_lm, kernel=kernel, forward=forward, budget=budget, run_dir=Path(run_dir),
                     work_dir=Path(work_dir), round_prompt=round_prompt, edit_store=edit_store, handoff_every=handoff_every,
                     require_explanations=require_explanations, max_turns=max_turns)
    forward.allow_validation = False
    for name in edit_store.names():  # the search starts from the original model
        (edit_store.dir / f"{name}.npy").unlink()
    seed_id = search.new_candidate("the original model, with no edits")
    try:
        res = optimize(
            seed_candidate={"edit": seed_id}, trainset=sorted(forward.practice), valset=sorted(forward.validation),
            adapter=search, custom_candidate_proposer=search.propose, candidate_selection_strategy="pareto",
            reflection_minibatch_size=settings.minibatch_size, skip_perfect_score=True, use_merge=False,
            stop_callbacks=[search.should_stop], callbacks=[search], seed=seed,
            run_dir=str(Path(run_dir) / "gepa_state"), raise_on_exception=False, display_progress_bar=False,
        )
    finally:
        forward.allow_validation = True
    if search.error is not None:
        logger.error("the GEPA search stopped early after %d rounds: %s", search.round, search.error)
    best = res.best_candidate["edit"]
    restore_candidate(search.cands / best, search.work_dir)
    answer_path = search.cands / best / "answer.md"
    report = {
        "kind": "gepa",
        "settings": asdict(settings),
        "rounds": search.round,
        "best": best,
        "best_val": search.val_scores.get(best),
        "candidates": [{"id": c, "parent": search.parent_of.get(c), "val": search.val_scores.get(c)}
                       for c in sorted(p.name for p in search.cands.iterdir())],
        "rounds_log": str(Path(run_dir) / "rounds.jsonl"),
        **({"error": f"{type(search.error).__name__}: {search.error}"} if search.error is not None else {}),
        **search._writer_usage(),
    }
    return search.edits_of(best), answer_path.read_text() if answer_path.exists() else "", report
