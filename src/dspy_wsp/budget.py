"""The writer's budget: forward passes and code runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Budget:
    max_passes: int = 50
    max_code_runs: int = 300
    passes_used: int = 0
    code_runs_used: int = 0

    @property
    def passes_left(self) -> int:
        return max(self.max_passes - self.passes_used, 0)

    @property
    def code_runs_left(self) -> int:
        return max(self.max_code_runs - self.code_runs_used, 0)

    def status(self) -> str:
        return f"[forward passes used: {self.passes_used} of {self.max_passes}; Python runs used: {self.code_runs_used} of {self.max_code_runs}]"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Budget":
        return cls(**{k: d[k] for k in ("max_passes", "max_code_runs", "passes_used", "code_runs_used") if k in d})
