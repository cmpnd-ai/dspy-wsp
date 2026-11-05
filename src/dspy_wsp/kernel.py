"""Host side of the writer's kernel: start it, run code, enforce time and output limits.

The paper ran the writer's code under bubblewrap with no network, a read-write
work folder and a time limit per run. `sandbox="auto"` (the default) does the
same when `bwrap` is installed and refuses to start otherwise; pass
`sandbox="none"` to run the writer's code with your own permissions.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dspy_wsp._kernel_main import MARK

logger = logging.getLogger(__name__)


class KernelError(RuntimeError):
    pass


def sandbox_prefix(mode: str, writable: list[Path], readonly: list[Path] = ()) -> list[str]:
    if mode == "none":
        return []
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise KernelError(
            "sandbox='auto' needs bubblewrap (`bwrap`) to isolate the writer's code. Install it "
            "(apt install bubblewrap), or pass sandbox='none' to run the writer's code with your own permissions and network."
        )
    cmd = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    # /tmp is replaced by an empty tmpfs: bind back what the kernel must read, read-only.
    for r in readonly:
        cmd += ["--ro-bind", str(r), str(r)]
    for w in writable:
        cmd += ["--bind", str(w), str(w)]
    return cmd + ["--unshare-all", "--die-with-parent", "--new-session"]


class Kernel:
    def __init__(
        self,
        *,
        work_dir: str | Path,
        model_dir: str | Path,
        tensor_index: dict[str, str],
        protected: tuple[str, ...],
        practice_dir: str | Path = "",
        timeout: float = 900,
        max_output: int = 10_000,
        memory_bytes: int | None = None,
        sandbox: str = "auto",
        threads: int | None = None,
        python: str | None = None,
        startup_timeout: float = 600,
        tokenizer_dir: str | Path | None = None,
    ):
        if sandbox not in ("auto", "none"):
            raise ValueError("sandbox must be 'auto' or 'none'")
        self.work_dir = Path(work_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.timeout = timeout
        self.max_output = max_output
        self.memory_bytes = memory_bytes
        self.sandbox = sandbox
        self.threads = threads
        self.python = python or sys.executable
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue | None = None
        self._tokenizer_dir = tokenizer_dir
        self._config_path = self.work_dir.parent / "kernel_config.json"
        self._config_path.write_text(
            json.dumps(
                {
                    "model_dir": str(self.model_dir),
                    "work_dir": str(self.work_dir),
                    "practice_dir": str(practice_dir),
                    "tensor_index": tensor_index,
                    "protected": list(protected),
                    "package_parent": str(Path(__file__).resolve().parent.parent),
                    **({"tokenizer_dir": str(Path(tokenizer_dir).resolve())} if tokenizer_dir else {}),
                }
            )
        )

    # --- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        script = Path(__file__).with_name("_kernel_main.py")
        readonly = [self.model_dir, self._config_path, script.parent] + ([Path(tokenizer_dir).resolve()] if (tokenizer_dir := self._tokenizer_dir) else [])
        cmd = sandbox_prefix(self.sandbox, [self.work_dir], readonly) + [self.python, "-I", "-u", str(script), str(self._config_path)]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/tmp" if self.sandbox != "none" else os.environ.get("HOME", "/tmp"),
            "LANG": "C.UTF-8",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if self.threads:
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(self.threads)

        mem = self.memory_bytes

        def _limits():  # runs in the child before exec
            if mem and sys.platform.startswith("linux"):
                import resource

                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.work_dir),
            start_new_session=True,
            preexec_fn=_limits,
            text=True,
            bufsize=1,
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()
        msg, stray = self._read_result(self.startup_timeout)
        if msg is None or not msg.get("ready"):
            detail = (msg or {}).get("error") or stray or "no response"
            self.close()
            raise KernelError(f"the writer's kernel failed to start:\n{detail}")

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue) -> None:
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _read_result(self, timeout: float) -> tuple[dict | None, str]:
        """Wait for one protocol line. Returns (message or None on timeout/exit, stray output)."""
        deadline = time.monotonic() + timeout
        stray: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "".join(stray)
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None, "".join(stray)
            if line is None:
                return {"exited": True}, "".join(stray)
            if line.startswith(MARK):
                return json.loads(line[len(MARK) :]), "".join(stray)
            stray.append(line)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self._proc.kill()
            self._proc.wait(timeout=30)
        self._proc = None

    def restart(self) -> None:
        self.close()
        self.start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # --- running code ----------------------------------------------------------------
    def run(self, code: str) -> str:
        if not self.alive:
            self.start()
        t0 = time.monotonic()
        self._proc.stdin.write(json.dumps({"code": code}) + "\n")
        self._proc.stdin.flush()
        msg, stray = self._read_result(self.timeout)
        if msg is None:
            self.restart()
            out = f"[timed out after {self.timeout:.0f} s; the kernel was restarted and its variables are lost]"
        elif msg.get("exited"):
            self.restart()
            out = stray + "\n[the kernel died (often the memory limit); it was restarted and its variables are lost]"
        else:
            out = stray + msg.get("out", "")
        out = out.rstrip("\n")
        if len(out) > self.max_output:
            out = out[: self.max_output] + f"\n[output cut at {self.max_output:,} characters]"
        logger.debug("kernel run took %.1f s", time.monotonic() - t0)
        return out or "(no output)"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
