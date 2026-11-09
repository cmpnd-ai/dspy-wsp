"""A short end-to-end run: a real student, a real writer, two forward passes.

This checks that the pieces work together. It is far too small to improve the
model. It runs on a laptop (CUDA, Apple MPS or CPU), downloads Qwen3.5-0.8B
(about 1.7 GB) on first use, and needs an API key for the writer.

    pip install -e ".[examples]"
    OPENAI_API_KEY=... python examples/smoke_test.py
"""

import json
import sys
from pathlib import Path

import dspy
import torch
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

from dspy_wsp import HFBackend, Limits, WeightSpace

RUN_DIR = Path("runs/smoke")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
backend = HFBackend(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    device=device,
    max_tokens=512,
    chat_template_kwargs={"enable_thinking": False},  # short answers keep each pass fast
)

writer = dspy.LM("openai/gpt-6-luna")

data = GSM8K()
trainset, valset = data.train[:16], data.dev[:8]

optimizer = WeightSpace(
    metric=gsm8k_metric,
    writer_lm=writer,
    backend=backend,
    max_passes=2,
    max_code_runs=20,
    limits=Limits(max_generations=8, max_generations_both=8),
    task_description="The program answers grade-school math word problems; the answer field must be the final number.",
    kernel_timeout=120,
    run_dir=RUN_DIR,
    resume=True,
)

compiled = optimizer.compile(dspy.Predict("question -> answer"), trainset=trainset, valset=valset)

report = compiled.wsp_report
print(compiled.wsp_patch.summary())
print("budget:", report["budget"])
print("validation:", report["validation"]["paired"])

compiled.save(str(RUN_DIR / "program.json"))
reloaded = dspy.Predict("question -> answer")
reloaded.load(str(RUN_DIR / "program.json"), allow_unsafe_lm_state=True)
print("reloaded answer:", reloaded(question=valset[0].question).answer, "| expected:", valset[0].answer)

checks = {
    "used both passes": report["budget"]["passes_used"] == 2,
    "final answer written": (RUN_DIR / "final_answer.md").exists(),
    "patch written": (RUN_DIR / "patch" / "manifest.json").exists(),
    "report written": json.loads((RUN_DIR / "report.json").read_text()) is not None,
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(checks.values()) else 1)
