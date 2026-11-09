import os

import numpy as np
import pytest

from dspy_wsp.edits import DEFAULT_PROTECTED
from dspy_wsp.kernel import Kernel, KernelError, find_sandbox
from dspy_wsp.tensors import TensorReader

L0 = "model.layers.0.mlp.down_proj.weight"


@pytest.fixture
def kernel(weights_dir, tmp_path):
    work = tmp_path / "run" / "work"
    work.mkdir(parents=True)
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index=TensorReader(weights_dir).index,
               protected=DEFAULT_PROTECTED, sandbox="none", timeout=20, max_output=2000)
    k.start()
    yield k
    k.close()


def test_state_persists_and_last_expression_shows(kernel):
    assert kernel.run("x = 41") == "(no output)"
    assert kernel.run("x + 1") == "42"
    assert kernel.run("print('hi'); len(NAMES)") == "hi\n4"


def test_errors_show_the_cell_traceback(kernel):
    out = kernel.run("1/0")
    assert "ZeroDivisionError" in out and "_kernel_main" not in out
    assert kernel.run("exit()") .startswith("Traceback") or "SystemExit" in kernel.run("raise SystemExit")
    assert kernel.run("2 + 2") == "4"  # the kernel survived


def test_save_edit_and_rules(kernel, weights_dir, tmp_path):
    out = kernel.run(f"w = load('{L0}'); w[:, 0] = 0; save_edit('{L0}', w)")
    assert "saved edit" in out
    assert kernel.run("edited()") == f"['{L0}']"
    assert kernel.run(f"float((load('{L0}')[:, 0]).abs().sum())") == "0.0"
    out = kernel.run("save_edit('model.norm.weight', load('model.norm.weight') + 1)")
    assert "out of scope" in out
    out = kernel.run(f"save_edit('{L0}', load_original('{L0}') * 2)")
    assert "multiplies the whole tensor" in out
    assert kernel.run("torch.is_grad_enabled()") == "False"
    saved = np.load(tmp_path / "run" / "work" / "edited" / f"{L0}.npy")
    assert (saved[:, 0] == 0).all()


def test_c_level_output_is_captured(kernel):
    assert "from os" in kernel.run("import os; os.write(1, b'from os\\n')")


def test_timeout_restarts_the_kernel(weights_dir, tmp_path):
    work = tmp_path / "w2" / "work"
    work.mkdir(parents=True)
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index=TensorReader(weights_dir).index,
               protected=DEFAULT_PROTECTED, sandbox="none", timeout=3)
    k.start()
    try:
        k.run("y = 5")
        out = k.run("import time; time.sleep(30)")
        assert "timed out" in out
        assert "NameError" in k.run("y")
    finally:
        k.close()


def test_output_is_cut(kernel):
    assert "output cut" in kernel.run("print('x' * 5000)")


def has_sandbox() -> bool:
    try:
        return find_sandbox("auto") is not None
    except KernelError:
        return False


def test_auto_sandbox_needs_bwrap_on_linux(weights_dir, tmp_path, monkeypatch):
    import shutil
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    work = tmp_path / "w3" / "work"
    work.mkdir(parents=True)
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index={}, protected=(), sandbox="auto")
    with pytest.raises(KernelError, match="bubblewrap"):
        k.start()


def test_optimizer_refuses_a_missing_sandbox_at_construction(monkeypatch):
    import shutil
    import sys

    from dspy_wsp import WeightSpace

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(KernelError, match="bubblewrap"):
        WeightSpace(metric=lambda *a: 1.0, writer_lm=None, backend=object(), sandbox="auto")
    with pytest.raises(ValueError, match="sandbox must be"):
        WeightSpace(metric=lambda *a: 1.0, writer_lm=None, backend=object(), sandbox="docker")
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(KernelError, match="no sandbox for win32"):
        WeightSpace(metric=lambda *a: 1.0, writer_lm=None, backend=object(), sandbox="auto")


def test_allowlist_keeps_parents_first_and_refuses_home(tmp_path):
    from pathlib import Path

    from dspy_wsp.kernel import _allowlist

    (tmp_path / "a" / "b").mkdir(parents=True)
    real = str(tmp_path.resolve())
    assert [r for r, _ in _allowlist([tmp_path / "a" / "b", tmp_path / "a", tmp_path / "missing"])] == [f"{real}/a"]
    assert _allowlist([tmp_path / "a"], covered=[real]) == []
    with pytest.raises(KernelError, match="will not expose"):
        _allowlist([Path.home()])


TRY = "try:\n    {}\n    print('OPEN')\nexcept OSError as e:\n    print('BLOCKED', type(e).__name__)"


@pytest.mark.skipif(not has_sandbox(), reason="no sandbox on this machine")
def test_sandbox_reads_an_allowlist_and_has_no_network(weights_dir, tmp_path):
    from pathlib import Path

    work = tmp_path / "sb" / "work"
    work.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    outside = tmp_path / "outside.txt"
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index=TensorReader(weights_dir).index,
               protected=DEFAULT_PROTECTED, sandbox="auto", timeout=60)
    k.start()
    try:
        assert "saved edit" in k.run(f"w = load('{L0}'); w[0, 0] += 1; save_edit('{L0}', w)")
        assert (work / "edited" / f"{L0}.npy").exists()
        assert "tmp ok" in k.run("import tempfile; f = tempfile.NamedTemporaryFile(); f.write(b'x'); print('tmp ok')")
        assert "BLOCKED" in k.run(TRY.format("import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"))
        for probe in (f"open({str(secret)!r}).read()", f"open({str(Path.home() / '.profile')!r}).read()"):
            out = k.run(TRY.format(probe))
            assert "BLOCKED" in out and "TOPSECRET" not in out, probe
        # The home folder is unreadable (macOS) or holds only the folders that lead to allowlisted paths (Linux).
        import sys

        from dspy_wsp.kernel import runtime_paths

        allowed = [str(Path(p).resolve()) for p in [*runtime_paths(), work, weights_dir, sys.modules["dspy_wsp"].__file__]]
        hidden = [n for n in os.listdir(Path.home()) if not any(a.startswith(str((Path.home() / n).resolve())) for a in allowed)]
        out = k.run(TRY.format(f"print(sorted(os.listdir({str(Path.home())!r})))"))
        assert "BLOCKED" in out or not any(repr(n) in out for n in hidden), out
        k.run(TRY.format(f"open({str(outside)!r}, 'w').write('x')"))  # refused, or it lands in the sandbox's own /tmp
        assert not outside.exists()
        # A link made inside the sandbox cannot read the file for the kernel, and the host refuses to follow it.
        out = k.run(f"os.symlink({str(secret)!r}, os.path.join(WORK, 'notes.md'))\n" + TRY.format("print(open(os.path.join(WORK, 'notes.md')).read())"))
        assert "BLOCKED" in out and "TOPSECRET" not in out
    finally:
        k.close()
    from dspy_wsp.edits import read_notes

    assert "TOPSECRET" not in read_notes(work) and "link" in read_notes(work)


def test_host_refuses_links_in_the_work_folder(weights_dir, tmp_path):
    import os

    from dspy_wsp.edits import EditError, EditStore, read_notes

    work = tmp_path / "work"
    store = EditStore(work / "edited", TensorReader(weights_dir))
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    assert read_notes(work) == ""
    (work / "notes.md").write_text("my notes")
    assert read_notes(work) == "my notes"
    os.link(secret, work / "hard.md")
    (work / "notes.md").unlink()
    os.rename(work / "hard.md", work / "notes.md")
    assert "TOPSECRET" not in read_notes(work)

    np.save(tmp_path / "elsewhere.npy", np.zeros((4, 8), np.float32))
    os.symlink(tmp_path / "elsewhere.npy", work / "edited" / f"{L0}.npy")
    with pytest.raises(EditError, match="link"):
        store.load_all()
    assert "INVALID" in store.describe()
