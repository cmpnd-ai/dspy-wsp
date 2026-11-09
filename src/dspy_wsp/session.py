"""The writer's session: one block per turn, output back as the next message.

Mirrors the paper's harness: the writer replies with an explanation and exactly
one ```python or ```forward block; the session runs it and sends the result.
Every `handoff_every` turns the conversation restarts with the task, the
writer's notebook and its current edit, which keeps the context short. Every
turn is written to disk, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

from dspy_wsp import prompts
from dspy_wsp.blocks import parse_reply
from dspy_wsp.budget import Budget
from dspy_wsp.edits import read_notes

logger = logging.getLogger(__name__)


WRITER_RETRIES = 2  # extra tries for a writer call that times out or runs past the output limit


class Session:
    def __init__(
        self,
        *,
        writer_lm,
        kernel,
        forward_tool,
        budget: Budget,
        run_dir: str | Path,
        work_dir: str | Path,
        task_prompt: str,
        edit_store,
        handoff_every: int = 40,
        require_explanations: bool = True,
        max_turns: int = 1000,
        max_nudges: int = 3,
        pass_offset: int = 0,
    ):
        self.writer_lm = writer_lm
        self.kernel = kernel
        self.forward = forward_tool
        self.budget = budget
        self.run_dir = Path(run_dir)
        self.work_dir = Path(work_dir)
        self.task_prompt = task_prompt
        self.edit_store = edit_store
        self.handoff_every = handoff_every
        self.require_explanations = require_explanations
        self.max_turns = max_turns
        self.max_nudges = max_nudges
        self.pass_offset = pass_offset  # numbers this session's passes after earlier sessions' in the same work folder
        self.turn = 0
        self.conversation = 1
        self.since_handoff = 0
        self.done = False
        self.final_answer = ""
        self.last_usage: dict | None = None
        self.writer_usage = {"calls": 0, "tokens_in": 0, "cached_in": 0, "tokens_out": 0, "reasoning_tokens": 0, "usd": 0.0}

    # --- persistence ----------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save_state(self) -> None:
        state = {
            "turn": self.turn,
            "conversation": self.conversation,
            "since_handoff": self.since_handoff,
            "done": self.done,
            "budget": self.budget.to_dict(),
            "writer_usage": self.writer_usage,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_path)

    def load_state(self) -> bool:
        if not self.state_path.exists():
            return False
        s = json.loads(self.state_path.read_text())
        self.turn, self.conversation, self.done = s["turn"], s["conversation"], s["done"]
        b = Budget.from_dict(s["budget"])
        self.budget.passes_used, self.budget.code_runs_used = b.passes_used, b.code_runs_used
        self.writer_usage.update(s.get("writer_usage") or {})
        final = self.run_dir / "final_answer.md"
        self.final_answer = final.read_text() if final.exists() else ""
        return True

    def _log(self, event: dict) -> None:
        event = {"t": round(time.time(), 1), "turn": self.turn, **event}
        with open(self.run_dir / "transcript.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")

    def _save_turn(self, prompt: str, answer: str) -> None:
        d = self.run_dir / "turns" / f"t{self.turn:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(prompt)
        (d / "answer.md").write_text(answer)
        if self.last_usage is not None:
            (d / "usage.json").write_text(json.dumps(self.last_usage, indent=1))

    # --- the writer -----------------------------------------------------------------
    def _ask(self, messages: list[dict]) -> str:
        from dspy.utils.exceptions import LMInvalidRequestError, LMTransportError

        for attempt in range(WRITER_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = self.writer_lm(messages=messages)
                break
            except (LMTransportError, LMInvalidRequestError) as e:
                # a timeout or dropped connection, or a reply cut off at the model's output limit
                retryable = isinstance(e, LMTransportError) or "output limit" in str(e)
                if not retryable or attempt == WRITER_RETRIES:
                    raise
                logger.warning("the writer's call failed (%s); retrying", e)
        self._record_usage()
        first = out[0] if out else ""
        if isinstance(first, dict):
            first = first.get("text") or ""
        return str(first)

    def _record_usage(self) -> None:
        """Keep the last writer call's tokens and cost, and add them to the run's total."""
        history = getattr(self.writer_lm, "history", None) or []
        if not history:
            self.last_usage = None
            return
        entry = history[-1]
        u = entry.get("usage") or {}
        hit = bool(getattr(entry.get("response"), "cache_hit", False))  # a cached reply keeps its old cost but bills nothing
        self.last_usage = {
            "cache_hit": hit,
            "tokens_in": u.get("prompt_tokens") or 0,
            "cached_in": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "tokens_out": u.get("completion_tokens") or 0,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "usd": 0.0 if hit else entry.get("cost") or 0.0,
        }
        total = self.writer_usage
        total["calls"] += 1
        for k, v in self.last_usage.items():
            if k == "cache_hit":
                continue
            total[k] = round(total[k] + v, 6) if k == "usd" else total[k] + v

    def _handoff(self, kernel_restarted: bool) -> str:
        return prompts.handoff(self.task_prompt, read_notes(self.work_dir), self.edit_store.describe(), self.budget.status(), kernel_restarted)

    # --- the loop -------------------------------------------------------------------
    def run(self, resume: bool = False) -> str:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume and self.load_state():
            if self.done:
                return self.final_answer
            self.conversation += 1
            next_user = self._handoff(kernel_restarted=True)
        else:
            next_user = self.task_prompt
        messages: list[dict] = []
        explain_strikes = nudges = 0

        while self.turn < self.max_turns:
            self.turn += 1
            msgs = messages + [{"role": "user", "content": next_user}]
            reply = self._ask(msgs)
            self._save_turn(next_user, reply)
            messages = msgs + [{"role": "assistant", "content": reply}]
            self.since_handoff += 1
            parsed = parse_reply(reply)
            b = self.budget
            out = ""

            if parsed.kind == "final":
                if b.passes_left == 0 or nudges >= self.max_nudges:
                    self.done, self.final_answer = True, parsed.body
                    (self.run_dir / "final_answer.md").write_text(parsed.body)
                    self._log({"kind": "final"})
                    self.save_state()
                    break
                nudges += 1
                out = (
                    f"The session continues until all {b.max_passes} forward passes are used ({b.passes_left} left); "
                    "replying without a code block does not end it. Reply with one ```python or ```forward block."
                )
            elif parsed.kind == "invalid":
                out = parsed.error
            elif self.require_explanations and not parsed.explanation and explain_strikes < 2:
                explain_strikes += 1
                out = "This block was not run: explain in plain English what it does and why before the block, then send it again."
            elif parsed.kind == "python":
                explain_strikes = nudges = 0
                if b.code_runs_left == 0:
                    out = f"No Python runs are left (all {b.max_code_runs} used). Use a forward block, or reply without a code block to finish."
                else:
                    b.code_runs_used += 1
                    result = self.kernel.run(parsed.body)
                    out = f"OUTPUT of Python run {b.code_runs_used} of at most {b.max_code_runs}:\n```\n{result}\n```"
                    self._log({"kind": "python", "code": parsed.body, "output": result[:6000]})
            elif parsed.kind == "forward":
                explain_strikes = nudges = 0
                if b.passes_left == 0:
                    out = f"No forward passes are left. Make your final choice: leave the edit you submit in {self.work_dir}/edited, then reply without a code block."
                else:
                    k = self.pass_offset + b.passes_used + 1
                    out, counted = self.forward.handle(parsed.body, k)
                    if counted:
                        b.passes_used += 1
                        if b.passes_left == 0:
                            out += f"\nThat was your last forward pass. Make your final choice: leave the edit you submit in {self.work_dir}/edited, then reply without a code block."
                    self._log({"kind": "forward", "request": parsed.body, "counted": counted, "output": out[:6000]})

            next_user = f"{out}\n{b.status()}"
            if self.since_handoff >= self.handoff_every:
                messages = []
                self.since_handoff = 0
                self.conversation += 1
                next_user = self._handoff(kernel_restarted=False) + "\n\n" + next_user
            self.save_state()
        else:
            self.final_answer = self.final_answer or "(the session reached max_turns without a final answer)"
            self.done = True
            self.save_state()
        return self.final_answer
