"""GEPA over edits programs: a population of networks, each written as an edits program, improved by a writer LM.

Each GEPA iteration picks a parent from the Pareto frontier over validation instances, evaluates it on a minibatch of
training instances with feedback, and asks the writer for one edits block. The child (parent program + block) is
evaluated on the same minibatch; if it scores higher it is evaluated on the validation set and joins the population.
The budget counts evaluation units (e.g. episodes), so runs compare at matched forward passes.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..writer import ask_writer
from . import language

PROTOCOL = (
    "\n\nHOW THIS WORKS: you are shown one network (its full edits program, applied to all-zero weights) and how it "
    "did on some evaluations. Reply with a brief explanation of what you change and why, then exactly one edits "
    "block. It is applied on top of the network shown; make its first line `# base: zero` to start from all-zero "
    "weights instead.")

# evaluate(weights, instances, with_feedback) -> (one score per instance, feedback text or None)
Evaluate = Callable[[dict, Sequence[Any], bool], tuple[Sequence[float], str | None]]


def program_id(program: str) -> str:
    return hashlib.sha1(program.encode()).hexdigest()[:10]


@dataclass
class EditGEPAResult:
    best_program: str
    best_weights: dict[str, np.ndarray]
    best_val: float
    candidates: list[dict]       # every program evaluated on the validation set: id, program, val, units_used
    units_used: int
    writer_calls: int
    writer_tokens: dict          # in, cached, out, reasoning
    writer_usd: float | None     # from `prices` if given, else the LM's reported cost if any
    seconds: float


class EditGEPA:
    """GEPA population search where a writer LM edits a small network's weights through the `edits` language.

    Args:
        writer_lm: a `dspy.LM` (or any callable taking `messages=` and returning a list of completions).
        shapes: tensor name -> shape; the names the edits language may assign to.
        evaluate: `evaluate(weights, instances, with_feedback)` returns one score per instance and, when
            `with_feedback` is true, the feedback text the writer reads (behaviour, traces, statistics).
        task: the writer's system prompt: the task, the network, and the edits language.
        budget: evaluation units for the whole search; each evaluated instance costs `units_per_instance`.
        context: text shown before every parent program (e.g. the seed network's design notes).
        protocol: appended to `task`; explains the reply format.
        minibatch_size: training instances per reflection minibatch.
        prices: (input, cached input, output) $ per million tokens, for cost accounting.
        run_dir: where proposals, evaluations, candidates and GEPA's state are written.
    """

    def __init__(self, writer_lm, shapes: Mapping[str, tuple[int, ...]], evaluate: Evaluate, task: str, budget: int,
                 units_per_instance: int = 1, context: str = "", protocol: str = PROTOCOL, minibatch_size: int = 8,
                 prices: tuple[float, float, float] | None = None, seed: int = 0, run_dir: str | Path | None = None,
                 attempts: int = 2):
        self.writer_lm, self.shapes, self.evaluate_fn = writer_lm, dict(shapes), evaluate
        self.task, self.context, self.protocol = task, context, protocol
        self.budget, self.upi, self.minibatch_size = budget, units_per_instance, minibatch_size
        self.prices, self.seed, self.attempts = prices, seed, attempts
        self.run_dir = Path(run_dir) if run_dir else None
        self.used, self.tokens, self.lm_cost = 0, Counter(), 0.0
        self._weights: dict[str, tuple[dict, list[str]]] = {}
        self._instances: dict[str, Sequence[Any]] = {}

    # --- helpers ---------------------------------------------------------------------
    def weights(self, program: str) -> tuple[dict[str, np.ndarray], list[str]]:
        if program not in self._weights:
            self._weights[program] = language.apply(program, self.shapes)
        return self._weights[program]

    def _log(self, name: str, obj: dict) -> None:
        if self.run_dir:
            with open(self.run_dir / f"{name}.jsonl", "a") as f:
                f.write(json.dumps(obj, default=str) + "\n")

    def usd(self) -> float | None:
        if self.prices:
            p_in, p_cached, p_out = self.prices
            t = self.tokens
            return (p_in * (t["in"] - t["cached"]) + p_cached * t["cached"] + p_out * t["out"]) / 1e6
        return self.lm_cost or None

    def _ask(self, messages: list[dict]) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reply = ask_writer(self.writer_lm, messages)
        entry = (getattr(self.writer_lm, "history", None) or [{}])[-1]
        usage = entry.get("usage") or {}
        get = lambda o, k: (o.get(k) if isinstance(o, Mapping) else getattr(o, k, None)) or 0  # noqa: E731
        self.tokens.update(calls=1, **{"in": get(usage, "prompt_tokens"), "out": get(usage, "completion_tokens"),
                                       "cached": get(get(usage, "prompt_tokens_details") or {}, "cached_tokens"),
                                       "reasoning": get(get(usage, "completion_tokens_details") or {},
                                                        "reasoning_tokens")})
        self.lm_cost += entry.get("cost") or 0.0
        return reply

    # --- the GEPA adapter --------------------------------------------------------------
    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        program = candidate["program"]
        weights, skipped = self.weights(program)
        split = batch[0][0] if batch else "train"
        scores, feedback = self.evaluate_fn(weights, [self._instances[s][i] for s, i in batch], capture_traces)
        scores = [float(s) for s in scores]
        self.used += len(batch) * self.upi
        mean = sum(scores) / max(1, len(scores))
        self._log("evals", {"program": program_id(program), "split": split, "instances": len(batch), "mean": mean,
                            "units_used": self.used})
        if split == "val":
            self._log("programs", {"program": program_id(program), "text": program, "val": mean,
                                   "units_used": self.used, "skipped": skipped})
        note = ("\n".join(skipped) + "\n") if skipped else ""
        return EvaluationBatch(outputs=scores, scores=scores,
                               trajectories=[note + (feedback or "")] if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {"program": [{"feedback": eval_batch.trajectories[0]}]}

    propose_new_texts = None

    def propose(self, candidate, reflective_dataset, components_to_update):
        parent = candidate["program"]
        feedback = reflective_dataset["program"][0]["feedback"]
        context = f"{self.context}\n\n" if self.context else ""
        messages = [{"role": "system", "content": self.task + self.protocol},
                    {"role": "user", "content": f"{context}THE NETWORK (its full edits program):\n```edits\n{parent}\n"
                                                f"```\n\nHOW IT DID:\n{feedback}"}]
        for _ in range(self.attempts):
            reply = self._ask(messages)
            block = language.last_block(reply)
            self._log("proposals", {"parent": program_id(parent), "feedback": feedback, "reply": reply,
                                    "tokens": dict(self.tokens), "usd_total": self.usd(), "units_used": self.used})
            if block:
                if language.base_of(block, "") == "zero":
                    return {"program": block.strip()}
                return {"program": f"{parent}\n# ---- edit ----\n{block.strip()}"}
        raise RuntimeError(f"no edits block in {self.attempts} replies")

    # --- the search --------------------------------------------------------------------
    def run(self, seed_program: str, trainset: Sequence[Any], valset: Sequence[Any]) -> EditGEPAResult:
        """Search from `seed_program`. GEPA reflects on `trainset` minibatches and selects on `valset`."""
        from gepa.api import optimize

        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._instances = {"train": list(trainset), "val": list(valset)}
        reserve = (2 * self.minibatch_size + len(valset)) * self.upi  # what one more iteration can cost
        t0 = time.time()
        res = optimize(
            seed_candidate={"program": seed_program},
            trainset=[("train", i) for i in range(len(trainset))], valset=[("val", i) for i in range(len(valset))],
            adapter=self, custom_candidate_proposer=self.propose, candidate_selection_strategy="pareto",
            reflection_minibatch_size=self.minibatch_size, skip_perfect_score=False, use_merge=False,
            stop_callbacks=[lambda gepa_state: self.used + reserve > self.budget], seed=self.seed,
            run_dir=str(self.run_dir / "gepa_state") if self.run_dir else None, raise_on_exception=False,
            display_progress_bar=False)
        best = res.best_candidate["program"]
        candidates = []
        if self.run_dir and (self.run_dir / "programs.jsonl").exists():
            candidates = [json.loads(line) for line in open(self.run_dir / "programs.jsonl")]
        out = EditGEPAResult(best_program=best, best_weights=self.weights(best)[0],
                             best_val=float(res.val_aggregate_scores[res.best_idx]), candidates=candidates,
                             units_used=self.used, writer_calls=self.tokens["calls"], writer_tokens=dict(self.tokens),
                             writer_usd=self.usd(), seconds=time.time() - t0)
        if self.run_dir:
            (self.run_dir / "best_program.txt").write_text(best)
            summary = {k: v for k, v in out.__dict__.items() if k not in ("best_program", "best_weights", "candidates")}
            (self.run_dir / "result.json").write_text(json.dumps(summary, indent=1))
        return out
