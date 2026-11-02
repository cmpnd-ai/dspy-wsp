"""The writer's persistent Python kernel. Runs in its own process, ideally sandboxed.

Started as:  python -I -u _kernel_main.py <config.json>

Protocol: the host writes one JSON object per line on stdin ({"code": "..."}).
The kernel answers each with one line on its original stdout, prefixed with MARK.
Everything the code prints (Python or C level) is captured through fd 1 and 2.

Defined for the writer (the paper's Experiment 8 API):
  NAMES, SHAPES, MODEL, WORK, PRACTICE
  load(name)           float32 torch tensor: the current edit if there is one, else the original
  load_original(name)  float32 torch tensor of the original
  save_edit(name, t)   save an edit (checked: shape, finite, out-of-scope tensors, whole-tensor rescaling)
  drop_edit(name)      remove an edit
  edited()             names of edited tensors
  tok                  the model's tokenizer (loaded on first use)
Gradients are disabled. The kernel holds tensors, not a model: running the
network is the forward tool's job.
"""

import ast
import json
import os
import sys
import tempfile
import traceback

MARK = "\x1e@@WSP@@"
MAX_REPR = 10_000


def _prelude(cfg: dict, ns: dict) -> None:
    if cfg.get("package_parent"):
        sys.path.insert(0, cfg["package_parent"])
    import numpy as np
    import torch
    from safetensors import safe_open

    from dspy_wsp import edits as _edits

    torch.set_grad_enabled(False)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    model_dir, work = cfg["model_dir"], cfg["work_dir"]
    edit_dir = os.path.join(work, "edited")
    os.makedirs(edit_dir, exist_ok=True)
    index = cfg["tensor_index"]
    protected = tuple(cfg["protected"])
    names = sorted(index)

    shapes = {}
    by_file = {}
    for n, f in index.items():
        by_file.setdefault(f, []).append(n)
    for f, ns_ in by_file.items():
        with safe_open(os.path.join(model_dir, f), framework="pt") as h:
            for n in ns_:
                shapes[n] = tuple(h.get_slice(n).get_shape())

    def _check_name(name):
        if name not in index:
            raise KeyError(f"unknown tensor {name!r}; see NAMES")

    def load_original(name):
        _check_name(name)
        with safe_open(os.path.join(model_dir, index[name]), framework="pt") as h:
            return h.get_tensor(name).float()

    def load(name):
        _check_name(name)
        p = _edits.edit_path(edit_dir, name)
        if p.exists():
            return torch.from_numpy(np.load(p))
        return load_original(name)

    def save_edit(name, tensor):
        _check_name(name)
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().float().cpu().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        s = _edits.check_edit(name, arr, load_original(name).numpy(), protected)
        _edits.save_array(edit_dir, name, arr)
        print(f"saved edit {name}: {s['changed_entries']:,} of {s['total_entries']:,} entries changed, relative change {s['relative_change']:.4f}")

    def drop_edit(name):
        _check_name(name)
        p = _edits.edit_path(edit_dir, name)
        if p.exists():
            p.unlink()
            print(f"dropped edit {name}")
        else:
            print(f"{name} had no edit")

    def edited():
        return sorted(p.name[: -len(".npy")] for p in __import__("pathlib").Path(edit_dir).glob("*.npy"))

    class _LazyTokenizer:
        _tok = None

        def _get(self):
            if self._tok is None:
                try:
                    from transformers import AutoTokenizer

                    self._tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                except Exception:
                    from tokenizers import Tokenizer

                    self._tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
            return self._tok

        def __getattr__(self, attr):
            return getattr(self._get(), attr)

        def __call__(self, *a, **k):
            return self._get()(*a, **k)

        def __repr__(self):
            return f"<tokenizer for {model_dir}, loaded on first use>"

    ns.update(
        np=np,
        torch=torch,
        os=os,
        json=json,
        NAMES=names,
        SHAPES=shapes,
        MODEL=model_dir,
        WORK=work,
        PRACTICE=cfg.get("practice_dir", ""),
        load=load,
        load_original=load_original,
        save_edit=save_edit,
        drop_edit=drop_edit,
        edited=edited,
        tok=_LazyTokenizer(),
    )


def _run(code: str, ns: dict) -> str:
    tmp = tempfile.TemporaryFile(mode="w+b")
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    try:
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                value = eval(compile(last, "<cell>", "eval"), ns)
                if value is not None:
                    r = repr(value)
                    print(r if len(r) <= MAX_REPR else r[:MAX_REPR] + " ...")
        except BaseException as e:  # SystemExit and KeyboardInterrupt must not end the kernel
            tb = e.__traceback__
            while tb is not None and tb.tb_frame.f_code.co_filename != "<cell>":
                tb = tb.tb_next
            traceback.print_exception(type(e), e, tb)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
    tmp.seek(0)
    return tmp.read().decode("utf-8", "replace")


def main() -> None:
    cfg = json.load(open(sys.argv[1]))
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    commands = os.fdopen(os.dup(0), "r")
    # The writer's code gets an empty stdin: exit() closes sys.stdin, and the
    # command channel must survive that.
    sys.stdin = open(os.devnull)
    os.dup2(sys.stdin.fileno(), 0)
    ns: dict = {"__name__": "__wsp_kernel__"}
    try:
        _prelude(cfg, ns)
        proto.write(MARK + json.dumps({"ready": True}) + "\n")
    except Exception:
        proto.write(MARK + json.dumps({"ready": False, "error": traceback.format_exc()}) + "\n")
        return
    for line in commands:
        if not line.strip():
            continue
        req = json.loads(line)
        out = _run(req["code"], ns)
        proto.write(MARK + json.dumps({"out": out}) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
