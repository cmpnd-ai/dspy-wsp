"""Host side of the writer's kernel: start it, run code, enforce time and output limits.

The paper ran the writer's code under bubblewrap with no network, a read-write
work folder and a time limit per run. `sandbox="auto"` (the default) does the
same with bubblewrap on Linux and with `sandbox-exec` on macOS, and it refuses
to start when neither is there. The sandbox is an allowlist: the kernel reads
the Python runtime, this package and the model's files, writes the work folder
and a scratch folder, and has no network. Pass `sandbox="none"` to run the
writer's code with your own permissions.
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
import sysconfig
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from dspy_wsp._kernel_main import MARK

logger = logging.getLogger(__name__)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# System folders a Linux process needs: the dynamic loader and shared libraries.
LINUX_SYSTEM = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
LINUX_SYSTEM_FILES = ("/etc/ld.so.cache", "/etc/alternatives")
# System folders a macOS process needs: dyld, the system frameworks and Homebrew's libraries.
MACOS_SYSTEM = ("/usr", "/bin", "/sbin", "/System", "/Library/Apple", "/opt/homebrew", "/private/var/db/dyld",
                "/private/var/db/timezone", "/private/etc/localtime", "/dev")

SEATBELT_PROFILE = """(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read-data (literal "/"))
(allow file-write-data (literal "/dev/null") (literal "/dev/dtracehelper"))
(allow file-ioctl (literal "/dev/null") (literal "/dev/dtracehelper"))
"""


class KernelError(RuntimeError):
    pass


def find_sandbox(mode: str) -> tuple[str, str] | None:
    """The sandbox for sandbox='auto' as (kind, program), or None for sandbox='none'. Raises if none is available."""
    if mode not in ("auto", "none"):
        raise ValueError("sandbox must be 'auto' or 'none'")
    if mode == "none":
        return None
    none = "pass sandbox='none' to run the writer's code with your own permissions and network"
    if sys.platform == "darwin":
        if os.access(SANDBOX_EXEC, os.X_OK):
            return "seatbelt", SANDBOX_EXEC
        raise KernelError(f"sandbox='auto' needs {SANDBOX_EXEC} to isolate the writer's code, and it is missing. Or {none}.")
    bwrap = shutil.which("bwrap") if sys.platform.startswith("linux") else None
    if bwrap is not None:
        return "bwrap", bwrap
    if sys.platform.startswith("linux"):
        raise KernelError(f"sandbox='auto' needs bubblewrap (`bwrap`) to isolate the writer's code. Install it (apt install bubblewrap), or {none}.")
    raise KernelError(f"sandbox='auto' has no sandbox for {sys.platform}: it uses bubblewrap on Linux and sandbox-exec on macOS. On this platform, {none}.")


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def runtime_paths(python: str | None = None) -> list[str]:
    """What a Python process must read to start: the interpreter, its standard library and its installed packages."""
    paths = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, os.path.realpath(python or sys.executable)]
    paths += [v for k, v in sysconfig.get_paths().items() if k in ("stdlib", "platstdlib", "purelib", "platlib")]
    return paths


def link_targets(folder: str | Path) -> list[str]:
    """Files outside `folder` that its symlinks point to (a Hugging Face snapshot links to its blobs)."""
    root = os.path.realpath(folder)
    out = set()
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                if os.path.exists(target) and not _inside(target, root):
                    out.add(target)
    return sorted(out)


def _allowlist(paths: Iterable[str | Path], covered: Iterable[str] = ()) -> list[tuple[str, str]]:
    """(real path, path as given) for every existing path, parents first, without paths inside another."""
    pairs = {}
    for p in paths:
        given = os.path.abspath(str(p))
        if os.path.exists(given):
            pairs.setdefault(os.path.realpath(given), given)
    for real in pairs:
        if real == "/" or real == os.path.realpath(Path.home()):
            raise KernelError(f"the sandbox will not expose {real}; give narrower folders")
    out = []
    for real in sorted(pairs, key=lambda r: (r.count("/"), r)):
        if not any(_inside(real, c) for c in list(covered) + [r for r, _ in out]):
            out.append((real, pairs[real]))
    return out


def _bwrap_prefix(bwrap: str, writable: list[tuple[str, str]], readonly: list[tuple[str, str]]) -> list[str]:
    cmd = [bwrap, "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc"]
    for folder in LINUX_SYSTEM:
        if os.path.islink(folder):  # a merged /usr: /lib is a link to usr/lib
            cmd += ["--symlink", os.readlink(folder), folder]
        elif os.path.isdir(folder):
            cmd += ["--ro-bind", folder, folder]
    for f in LINUX_SYSTEM_FILES:
        if os.path.exists(f):
            cmd += ["--ro-bind", f, f]
    for flag, pairs in (("--ro-bind", readonly), ("--bind", writable)):
        for real, given in pairs:
            cmd += [flag, real, real]
            if given != real:  # reached through a symlink: the kernel is started by that path
                cmd += [flag, real, given]
    return cmd + ["--unshare-all", "--die-with-parent", "--new-session"]


def _seatbelt_prefix(program: str, writable: list[tuple[str, str]], readonly: list[tuple[str, str]], listable: list[str]) -> list[str]:
    # The profile takes each path as a parameter, so no path is quoted into the profile's text.
    params, lines = [], []
    for i, folder in enumerate(listable):
        params += ["-D", f"L{i}={folder}"]
        lines.append(f'(allow file-read-data (literal (param "L{i}")))')
    for i, (real, _) in enumerate(readonly + writable):
        params += ["-D", f"P{i}={real}"]
        rule = "subpath" if os.path.isdir(real) else "literal"
        lines.append(f'(allow file-read* ({rule} (param "P{i}")))')
        if i >= len(readonly):
            lines.append(f'(allow file-write* ({rule} (param "P{i}")))')
    return [program, "-p", SEATBELT_PROFILE + "\n".join(lines) + "\n", *params]


def sandbox_prefix(mode: str, writable: list[Path], readonly: Iterable[str | Path] = (), python: str | None = None,
                   listable: Iterable[str | Path] = ()) -> list[str]:
    """The command prefix that runs a Python process in the sandbox ([] for sandbox='none').

    The process may write `writable`. It may read `readonly`, the Python runtime
    and the system's libraries. It may list the folders in `listable` without
    reading their files. It has no network.
    """
    found = find_sandbox(mode)
    if found is None:
        return []
    kind, program = found
    system = [os.path.realpath(p) for p in (MACOS_SYSTEM if kind == "seatbelt" else LINUX_SYSTEM) if os.path.exists(p)]
    rw = _allowlist(writable)
    ro = _allowlist(list(readonly) + runtime_paths(python), covered=system if kind == "bwrap" else ())
    if kind == "bwrap":
        return _bwrap_prefix(program, rw, ro)
    ro = _allowlist(system) + [p for p in ro if not any(_inside(p[0], s) for s in system)]
    return _seatbelt_prefix(program, rw, ro, [os.path.realpath(p) for p in listable])


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
        readonly: Iterable[str | Path] = (),
    ):
        if sandbox not in ("auto", "none"):
            raise ValueError("sandbox must be 'auto' or 'none'")
        self.readonly = [str(p) for p in readonly]  # more folders the sandboxed kernel may read
        self._scratch: str | None = None
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
        models = [self.model_dir] + ([Path(self._tokenizer_dir).resolve()] if self._tokenizer_dir else [])
        readonly = [*models, self._config_path, script.parent, *self.readonly] + [t for m in models for t in link_targets(m)]
        writable, home = [self.work_dir], os.environ.get("HOME", "/tmp")
        found = find_sandbox(self.sandbox)
        if found is not None:
            home = "/tmp"  # bubblewrap gives the kernel an empty /tmp of its own
            if found[0] == "seatbelt":
                home = self._scratch = os.path.realpath(tempfile.mkdtemp(prefix="wsp-kernel-"))
                writable.append(Path(home))
        # Python lists the package's parent folder to import the package.
        cmd = sandbox_prefix(self.sandbox, writable, readonly, self.python, listable=[script.parent.parent]) + [self.python, "-I", "-u", str(script), str(self._config_path)]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": home,
            **({"TMPDIR": home} if self._scratch else {}),
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
            hint = ""
            if find_sandbox(self.sandbox) is not None:
                hint = ("\nThe sandbox lets the kernel read only the Python runtime, this package and the model's files. "
                        "If a library lives elsewhere, name its folder in sandbox_readonly=[...].")
            raise KernelError(f"the writer's kernel failed to start:\n{detail}{hint}")

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
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

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
