"""WeightSpace: weight-space programming as a DSPy optimizer."""

from __future__ import annotations

import datetime
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import dspy
from dspy.teleprompt.teleprompt import Teleprompter

from dspy_wsp import prompts
from dspy_wsp.backends.base import WeightBackend
from dspy_wsp.budget import Budget
from dspy_wsp.edits import DEFAULT_PROTECTED, EditStore, is_protected
from dspy_wsp.forward import ForwardTool, Limits, _jsonable, copy_program
from dspy_wsp.kernel import Kernel
from dspy_wsp.patch import Patch
from dspy_wsp.session import Session
from dspy_wsp.tensors import TensorReader

logger = logging.getLogger(__name__)


class WeightSpace(Teleprompter):
    """Improve a DSPy program by having a frontier LM edit the student model's weights directly.

    The writer LM reads the weights in a sandboxed Python kernel, runs the program
    on practice examples through a budgeted forward-pass tool, and saves targeted
    weight edits. No gradients and no training: the writer reasons about what
    each part of the network computes. Method: Takehi, Teoh, Zhang and Khattab,
    "Sparks of Weight Space Programming" (2026).

    Args:
        metric: a DSPy metric, metric(example, prediction, trace=None) -> float, bool,
            or dspy.Prediction(score=..., feedback=...). Feedback is shown to the writer.
        writer_lm: the dspy.LM that reads and edits the weights. The paper used a frontier
            model at maximum reasoning effort.
        backend: a WeightBackend that owns the student model's weights (e.g. HFBackend).
        max_passes: forward-pass budget (the paper: 10, 25 or 50).
        max_code_runs: Python runs allowed in the writer's kernel (the paper: 300).
        task_description: optional sentence of context for the writer.
        run_dir: where turns, the writer's work folder, the patch and the report go.
            Reusing a run_dir with resume=True continues an interrupted run.
        limits: per-pass limits (defaults follow the paper).
        protected: regexes of tensor names the writer may read but not edit. None uses the
            backend's own list if it has one (decision models), else the input embedding,
            the output layer and the final norm.
        holdout: trainset examples held out for validation (an int, or a share below 1).
            The writer can score edits on them but sees only aggregate results; supported
            for decision backends.
        holdout_key: groups for a stratified holdout, e.g. lambda ex: ex.workflow.
        sandbox: "auto" isolates the writer's code with bubblewrap (required);
            "none" runs it with your permissions and network.
        kernel_timeout, kernel_memory_bytes, kernel_threads: limits for each Python run.
        handoff_every: restart the writer's conversation every N turns.
        require_explanations: send back blocks that come without a plain-English explanation.
        max_turns: hard stop for the session.
        seed: base seed for the per-item sampling seeds.
        val_samples: samples per validation example in the final check.
        fallback_if_worse: if the final check on valset shows a negative gain, return
            the original weights (an addition to the paper's protocol, off by default).
        resume: continue the run in run_dir if one is there.
    """

    def __init__(
        self,
        metric: Callable,
        writer_lm,
        backend: WeightBackend,
        *,
        max_passes: int = 50,
        max_code_runs: int = 300,
        task_description: str | None = None,
        run_dir: str | Path | None = None,
        limits: Limits | None = None,
        protected: tuple[str, ...] | None = None,
        sandbox: str = "auto",
        kernel_timeout: float = 900,
        kernel_memory_bytes: int | None = None,
        kernel_threads: int | None = None,
        max_output: int = 10_000,
        handoff_every: int = 40,
        require_explanations: bool = True,
        max_turns: int = 1000,
        seed: int = 0,
        val_samples: int = 1,
        fallback_if_worse: bool = False,
        resume: bool = False,
        holdout: int | float = 0,
        holdout_key: Callable | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.writer_lm = writer_lm
        self.backend = backend
        self.max_passes = max_passes
        self.max_code_runs = max_code_runs
        self.task_description = task_description
        self.run_dir = Path(run_dir) if run_dir else None
        self.limits = limits or Limits()
        if protected is None:
            default = getattr(backend, "default_protected", None)
            protected = default() if callable(default) else DEFAULT_PROTECTED
        self.protected = tuple(protected)
        self.holdout = holdout
        self.holdout_key = holdout_key
        self.decision = getattr(backend, "kind", "llm") == "decision"
        if holdout and not self.decision:
            raise ValueError("holdout is supported for decision backends only")
        self.sandbox = sandbox
        self.kernel_timeout = kernel_timeout
        self.kernel_memory_bytes = kernel_memory_bytes
        self.kernel_threads = kernel_threads
        self.max_output = max_output
        self.handoff_every = handoff_every
        self.require_explanations = require_explanations
        self.max_turns = max_turns
        self.seed = seed
        self.val_samples = val_samples
        self.fallback_if_worse = fallback_if_worse
        self.resume = resume

    def compile(self, student: dspy.Module, *, trainset: list[dspy.Example], teacher=None, valset: list[dspy.Example] | None = None, **kwargs) -> dspy.Module:
        """Run a weight-space session and return a copy of `student` whose LM runs the edited weights.

        trainset becomes the writer's practice set. valset, if given, is never shown to
        the writer; it is used once at the end for a paired original-vs-edited check.
        The returned program carries `.wsp_patch` (a Patch) and `.wsp_report` (a dict).
        """
        if teacher is not None:
            logger.warning("WeightSpace ignores `teacher`")
        if not trainset:
            raise ValueError("WeightSpace needs a non-empty trainset (the writer's practice examples)")
        for ex in trainset:
            if not ex.inputs().keys():
                raise ValueError("every trainset example needs input fields: build them with .with_inputs(...)")

        run_dir = (self.run_dir or Path("wsp_runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
        work = run_dir / "work"
        practice_dir = work / "practice"
        practice_dir.mkdir(parents=True, exist_ok=True)

        program = copy_program(student, self.backend.lm())

        practice = {}
        for i, ex in enumerate(trainset):
            pid = str(ex.get("id", f"train_{i}")) if "id" not in ex.inputs().keys() else f"train_{i}"
            if pid in practice:
                pid = f"train_{i}"
            practice[pid] = ex
        validation = self._split_holdout(practice) if self.holdout else {}
        practice_file = practice_dir / "trainset.jsonl"
        with open(practice_file, "w") as f:
            for pid, ex in practice.items():
                f.write(json.dumps({"id": pid, "inputs": {k: _jsonable(v) for k, v in ex.inputs().items()}, "labels": {k: _jsonable(v) for k, v in ex.labels().items() if k != "id"}}) + "\n")

        reader = TensorReader(self.backend.weights_dir)
        editable = set(self.backend.editable_names())
        protected = self.protected + tuple(f"^{re.escape(n)}$" for n in reader.names if n not in editable)
        edit_store = EditStore(work / "edited", reader, protected)
        budget = Budget(max_passes=self.max_passes, max_code_runs=self.max_code_runs)
        forward = ForwardTool(
            backend=self.backend, program=program, metric=self.metric, practice=practice,
            edit_store=edit_store, work_dir=work, limits=self.limits, seed=self.seed, validation=validation,
        )
        first = next(iter(practice.values()))
        shapes = reader.shapes()
        if self.decision:  # list only the loaded network's tensors (d1 checkpoints also hold unused media encoders)
            shapes = {n: shape for n, shape in shapes.items() if n in editable}
        build = prompts.build_decision_task_prompt if self.decision else prompts.build_task_prompt
        extra = {"n_validation": len(validation)} if self.decision else {}
        task_prompt = build(
            **extra,
            model_description=self.backend.describe(),
            task_description=self.task_description,
            program_description=prompts.describe_program(program),
            metric_description=prompts.describe_metric(self.metric),
            sampling=self._sampling(),
            protected_description=getattr(self.backend, "protected_description", None)
            or "the tensors marked (out of scope) below (the input embedding, the output layer and the final norm)",
            kernel_environment="no internet, no GPU" if self.sandbox != "none" else "no GPU",
            model_dir=str(Path(self.backend.weights_dir).resolve()),
            practice_dir=str(practice_dir),
            practice_file=str(practice_file),
            work_dir=str(work),
            timeout=int(self.kernel_timeout),
            max_output=self.max_output,
            max_code_runs=self.max_code_runs,
            tensor_listing=prompts.tensor_listing(shapes, protected=lambda n: is_protected(n, protected)),
            n_practice=len(practice),
            input_keys=sorted(first.inputs().keys()),
            label_keys=sorted(k for k in first.labels().keys() if k != "id"),
            max_passes=self.max_passes,
            limits=self.limits,
            n_layers=self.backend.dims().get("layers"),
            require_explanations=self.require_explanations,
        )
        (run_dir / "task_prompt.txt").write_text(task_prompt)

        kernel = Kernel(
            work_dir=work, model_dir=self.backend.weights_dir, tensor_index=reader.index, protected=protected,
            practice_dir=practice_dir, timeout=self.kernel_timeout, max_output=self.max_output,
            tokenizer_dir=getattr(self.backend, "tokenizer_dir", None) if self.decision else None,
            memory_bytes=self.kernel_memory_bytes, sandbox=self.sandbox, threads=self.kernel_threads,
        )
        session = Session(
            writer_lm=self.writer_lm, kernel=kernel, forward_tool=forward, budget=budget, run_dir=run_dir,
            work_dir=work, task_prompt=task_prompt, edit_store=edit_store, handoff_every=self.handoff_every,
            require_explanations=self.require_explanations, max_turns=self.max_turns,
        )
        try:
            kernel.start()
            final_answer = session.run(resume=self.resume)
        finally:
            kernel.close()

        edits = edit_store.load_all()
        report = {
            "run_dir": str(run_dir),
            "final_answer": final_answer,
            "budget": budget.to_dict(),
            "turns": session.turn,
            "conversations": session.conversation,
        }
        if self.decision:
            report["validation_runs"] = forward.validation_log
            report["validation_size"] = len(validation)
        if valset:
            check = forward.paired_eval(valset, edits, n=self.val_samples, out_dir=run_dir / "final_check")
            report["validation"] = check
            gain = check["paired"]["gain"]
            if self.fallback_if_worse and edits and gain < 0:
                logger.warning("the edit lowered the validation score by %.4f; returning the original weights", -gain)
                report["fallback"] = f"validation gain {gain:.4f} < 0; original weights kept"
                edits = {}
        patch = Patch.from_edits(
            reader, edits,
            meta={
                "base_model": self.backend.model_id,
                "revision": getattr(self.backend, "revision", None),
                "writer": getattr(self.writer_lm, "model", str(self.writer_lm)),
                "budget": budget.to_dict(),
                "run_dir": str(run_dir),
                "validation": report.get("validation", {}).get("paired"),
            },
        )
        patch.save(run_dir / "patch")
        report["patch"] = patch.summary()
        (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))

        self.backend.set_weights(edits)
        if hasattr(self.backend, "patch_dir"):
            self.backend.patch_dir = str(run_dir / "patch") if edits else None
        program._compiled = True
        program.wsp_patch = patch
        program.wsp_report = report
        return program

    def _split_holdout(self, practice: dict) -> dict:
        """Move `holdout` examples (stratified by `holdout_key`) from practice to validation, in place."""
        import random

        n_total = len(practice)
        n = int(round(self.holdout * n_total)) if isinstance(self.holdout, float) and self.holdout < 1 else int(self.holdout)
        if not 0 < n < n_total:
            raise ValueError(f"holdout must leave both practice and validation examples; got {n} of {n_total}")
        groups: dict = {}
        for pid, ex in practice.items():
            groups.setdefault(self.holdout_key(ex) if self.holdout_key else "", []).append(pid)
        keys = sorted(groups, key=str)
        quota = {k: n * len(groups[k]) / n_total for k in keys}
        take = {k: int(quota[k]) for k in keys}
        for k in sorted(keys, key=lambda k: quota[k] - take[k], reverse=True)[: n - sum(take.values())]:
            take[k] += 1  # the largest remainders get the leftover examples, so the total is exactly n
        rng = random.Random(self.seed)
        chosen = []
        for key in keys:
            ids = sorted(groups[key])
            rng.shuffle(ids)
            chosen += ids[: take[key]]
        validation = {pid: practice.pop(pid) for pid in sorted(chosen)}
        return validation

    def _sampling(self) -> str:
        b = self.backend
        parts = []
        for attr, label in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("max_tokens", "at most {} new tokens per LM call")):
            v = getattr(b, attr, None)
            if v is not None:
                parts.append(label.format(v) if "{}" in label else f"{label} {v}")
        ctk = getattr(b, "chat_template_kwargs", None)
        if ctk:
            parts.append(f"chat template options {ctk}")
        return ", ".join(parts) or "the backend's defaults"
