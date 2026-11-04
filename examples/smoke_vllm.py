"""Check a GPU machine before the full MATH-500 run: sandbox, weight swaps and speed.

    cd examples
    ../.venv/bin/python smoke_vllm.py            # 64 MATH-500 problems
    ../.venv/bin/python smoke_vllm.py --n 128 --gpu-mem 0.8

1. Sandbox: the writer's kernel runs under bubblewrap. The check confirms that code
   inside can write its work folder but not the rest of the disk, and has no network.
2. Weight swap: vLLM's prompt log-probs must match the Hugging Face model's on the
   original weights, on edited weights, and again after restoring. The edit touches a
   fused layer (gate_proj) and an unfused one (down_proj).
3. Speed: the unedited model answers n problems with the full run's settings. The
   report gives the score, the share cut off at the token limit and tokens per second.

No writer runs, so no API key is needed. Results go to runs/smoke_vllm.json.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import dspy
import numpy as np
from math500_qwen_mlx import BoxedAnswerAdapter, MathQA, math500, metric
from math500_qwen_vllm import student

from dspy_wsp.forward import ForwardTool
from dspy_wsp.kernel import sandbox_prefix

OUT = Path("runs/smoke_vllm.json")
TEXTS = [
    "The derivative of x^3 is 3x^2, so the slope of y = x^3 at x = 2 is 12.",
    "Let n be the smallest positive integer such that n is divisible by 6 and 8. Then n = 24.",
    "In a right triangle with legs 3 and 4, the hypotenuse has length 5.",
    "The sum of the interior angles of a hexagon is 720 degrees.",
]


def check_sandbox() -> dict:
    probe = """
import socket, pathlib, sys
work = pathlib.Path(sys.argv[1])
(work / "ok.txt").write_text("ok")
res = {"work_writable": (work / "ok.txt").exists()}
try:
    pathlib.Path.home().joinpath("wsp_sandbox_probe.txt").write_text("x"); res["home_writable"] = True
except OSError:
    res["home_writable"] = False
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3); res["network"] = True
except OSError:
    res["network"] = False
print(res)
"""
    with tempfile.TemporaryDirectory(dir=Path.home()) as work:
        cmd = sandbox_prefix("auto", [Path(work)]) + [sys.executable, "-c", probe, work]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return {"ok": False, "error": out.stderr.strip()[-500:]}
    res = ast.literal_eval(out.stdout.strip())
    res["ok"] = res["work_writable"] and not res["home_writable"] and not res["network"]
    return res


def vllm_prompt_logprobs(backend, texts: list[str]) -> list[np.ndarray]:
    """Log-prob of each token given the ones before it (position 0 is NaN), from vLLM."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)

    async def one(ids):
        final = None
        async for out in backend._engine.generate({"prompt_token_ids": ids}, params, request_id=f"smoke-{time.time_ns()}"):
            final = out
        return [np.nan] + [float(d[t].logprob) for d, t in zip(final.prompt_logprobs[1:], ids[1:])]

    return [np.array(backend._run(one(backend._encode(t))), np.float32) for t in texts]


def hf_logprobs(backend, texts: list[str]) -> list[np.ndarray]:
    arrays, _ = backend.teacher_forced(texts, token_logprobs=True)
    return [arrays["token_logprobs"][i, : arrays["lengths"][i]] for i in range(len(texts))]


def mean_gap(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return float(np.nanmean(np.concatenate([np.abs(x - y) for x, y in zip(a, b)])))


def check_swap(backend) -> dict:
    names = backend.editable_names()
    pick = [n for n in names if re.search(r"layers\.12\.mlp\.(down|gate)_proj\.weight$", n)]
    rng = np.random.default_rng(0)
    edits = {}
    for n in pick:
        w = backend.reader.get_f32(n)
        edits[n] = w + rng.normal(0, w.std(), size=w.shape).astype(np.float32)

    v0, h0 = vllm_prompt_logprobs(backend, TEXTS), hf_logprobs(backend, TEXTS)
    t = time.time()
    backend.set_weights(edits)
    swap_seconds = time.time() - t
    v1, h1 = vllm_prompt_logprobs(backend, TEXTS), hf_logprobs(backend, TEXTS)
    backend.set_weights(None)
    v2 = vllm_prompt_logprobs(backend, TEXTS)
    res = {
        "edited_tensors": pick,
        "vllm_vs_hf_original": round(mean_gap(v0, h0), 4),
        "vllm_vs_hf_edited": round(mean_gap(v1, h1), 4),
        "edit_effect_vllm": round(mean_gap(v0, v1), 4),
        "edit_effect_hf": round(mean_gap(h0, h1), 4),
        "vllm_after_restore_vs_before": round(mean_gap(v0, v2), 4),
        "swap_seconds": round(swap_seconds, 1),
    }
    agree = max(res["vllm_vs_hf_original"], res["vllm_vs_hf_edited"])
    res["ok"] = len(pick) == 2 and res["edit_effect_vllm"] > 5 * agree and res["vllm_after_restore_vs_before"] <= agree
    return res


def check_speed(backend, n: int) -> dict:
    items = [(f"val_{i}", ex) for i, ex in enumerate(math500(n))]
    tool = ForwardTool(backend=backend, program=dspy.Predict(MathQA), metric=metric, practice={}, edit_store=None, work_dir=OUT.parent / "smoke_vllm")
    t = time.time()
    rows = tool.run_items(tool._program(None), items, 1, "original")
    summary = ForwardTool.summarize(rows)
    summary["seconds"] = round(time.time() - t)
    summary["tokens_per_second"] = round(sum(r["output_tokens"] for r in rows) / max(1, summary["seconds"]))
    ForwardTool._write_rows(OUT.parent / "smoke_vllm" / "rows.jsonl", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64, help="MATH-500 problems for the speed check")
    parser.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM's share of GPU memory")
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    def save(stage):
        OUT.write_text(json.dumps(report, indent=1))
        print(f"--- {stage}\n{json.dumps(report[stage], indent=1)}", flush=True)

    report["sandbox"] = check_sandbox()
    save("sandbox")
    backend = student(args.gpu_mem)
    report["swap"] = check_swap(backend)
    save("swap")
    report["speed"] = check_speed(backend, args.n)
    save("speed")
    passed = report["sandbox"]["ok"] and report["swap"]["ok"]
    print("PASS" if passed else "FAIL", f"report: {OUT}")
    backend.close()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    dspy.configure(adapter=BoxedAnswerAdapter(use_json_adapter_fallback=False))
    main()
