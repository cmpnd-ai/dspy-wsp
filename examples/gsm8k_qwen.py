"""Edit Qwen3.5-0.8B's weights to improve a DSPy GSM8K program.

A scaled-down version of the paper's Experiment 8. Needs a GPU for the student
model, bubblewrap for the sandbox, and an API key for the writer.

    pip install -e ".[examples]"
    python examples/gsm8k_qwen.py
"""

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

from dspy_wsp import HFBackend, Limits, WeightSpace

# The student: thinking on, as in the paper. HFBackend generates one sequence at a
# time, so this keeps answers short; a batched vLLM backend is the way to full scale.
backend = HFBackend(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",  # pin the base so the patch names it exactly
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    chat_template_kwargs={"enable_thinking": True},
)

# The writer: any capable dspy.LM. The paper used a frontier model at maximum reasoning effort.
writer = dspy.LM("openai/gpt-6-luna")

data = GSM8K()
trainset, valset = data.train[:2000], data.dev[:200]

program = dspy.Predict("question -> answer")

optimizer = WeightSpace(
    metric=gsm8k_metric,
    writer_lm=writer,
    backend=backend,
    max_passes=25,
    limits=Limits(max_generations=128, max_generations_both=64),  # smaller passes for the in-process backend
    task_description="The program answers grade-school math word problems; the answer field must be the final number.",
    run_dir="runs/gsm8k-qwen",
    resume=True,  # rerun the script to continue after an interruption
)

compiled = optimizer.compile(program, trainset=trainset, valset=valset)

print(compiled.wsp_patch.summary())
print(compiled.wsp_report["validation"]["paired"])
compiled.save("runs/gsm8k-qwen/program.json")
