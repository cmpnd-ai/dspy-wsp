import numpy as np
import pytest

from dspy_wsp.edits import DEFAULT_PROTECTED
from dspy_wsp.kernel import Kernel, KernelError
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


def test_auto_sandbox_needs_bwrap(weights_dir, tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    work = tmp_path / "w3" / "work"
    work.mkdir(parents=True)
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index={}, protected=(), sandbox="auto")
    with pytest.raises(KernelError, match="bubblewrap"):
        k.start()


@pytest.mark.skipif(__import__("shutil").which("bwrap") is None, reason="bubblewrap not installed")
def test_bwrap_sandbox_blocks_network_and_outside_writes(weights_dir, tmp_path):
    work = tmp_path / "sb" / "work"
    work.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    k = Kernel(work_dir=work, model_dir=weights_dir, tensor_index=TensorReader(weights_dir).index,
               protected=DEFAULT_PROTECTED, sandbox="auto", timeout=30)
    k.start()
    try:
        net = k.run("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=3); print('NET OPEN')\nexcept OSError as e:\n    print('NET BLOCKED', type(e).__name__)")
        assert "NET BLOCKED" in net
        k.run(f"open({str(outside)!r}, 'w').write('x')")  # lands in the sandbox's throwaway /tmp
        assert not outside.exists()
        probe = __import__("pathlib").Path(__file__).with_name("sandbox_probe.txt")
        out = k.run(f"open({str(probe)!r}, 'w').write('x')")
        assert "Read-only file system" in out and not probe.exists()
        assert "saved edit" in k.run(f"w = load('{L0}'); w[0, 0] += 1; save_edit('{L0}', w)")
        assert (work / "edited" / f"{L0}.npy").exists()
    finally:
        k.close()
