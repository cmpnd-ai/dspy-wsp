"""Edit Qwen3.5-0.8B's weights to improve MATH-500, on a Mac.

This follows the paper's Experiment 8 setup on one Apple-silicon machine. The student
runs with thinking on and the model card's sampling (temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.5, up to 32,768 tokens). The writer practices on the
paper's 7,275 MATH train problems. The final check uses MATH-500, which the writer
never sees.

    pip install -e ".[examples]"
    python examples/math500_qwen_mlx.py --baseline 64   # the unedited model only; no writer, no API cost
    OPENAI_API_KEY=... python examples/math500_qwen_mlx.py

Long thinking makes this slow. At 32k tokens most unedited answers loop until the
limit, so a 64-problem "both" pass takes one to two hours on an M2 Max, and the
200-problem final check takes several hours. Rerun the script to resume.
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

import dspy

from dspy_wsp import Limits, MLXBackend, WeightSpace

RUN_DIR = Path("runs/math500-qwen-mlx")
PRACTICE_URL = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/practice/math_train.jsonl"
VAL_SIZE = 200  # MATH-500 problems in the final check; the paper used all 500 with 4 samples each
MAX_PASSES = 25

INSTRUCTIONS = "Solve the math problem. Please reason step by step, and put your final answer within \\boxed{}."


class MathQA(dspy.Signature):
    __doc__ = INSTRUCTIONS
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final answer, in \\boxed{}")


def metric(example, pred, trace=None):
    """Graded with math-verify, as in the paper's evaluation."""
    from math_verify import parse, verify

    # math-verify's default timeout uses signal.alarm, which fails outside the main thread.
    gold = parse(f"${example.answer}$", parsing_timeout=None)
    got = str(pred.answer)
    return float(any(verify(gold, parse(text, parsing_timeout=None), timeout_seconds=None) for text in (got, f"${got}$")))


def last_boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...} in text, with nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


class BoxedAnswerAdapter(dspy.ChatAdapter):
    """ChatAdapter that falls back to the last \\boxed{} answer when the reply lacks DSPy's field markers."""

    def parse(self, signature, completion):
        try:
            return super().parse(signature, completion)
        except dspy.utils.exceptions.AdapterParseError:
            answer = last_boxed(completion)
            if answer is None:
                raise
            return {"answer": answer}


class PaperPromptAdapter(dspy.Adapter):
    """The paper's evaluation protocol: one user turn holding the problem and the instruction,
    graded by the last \\boxed{} anywhere in the reply, thinking included.

    Use it with the backend's split_thinking=False, so answers cut off while thinking still reach the parser.
    """

    instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    def format(self, signature, demos, inputs):
        return [{"role": "user", "content": f"{inputs['question']}\n{self.instruction}"}]

    def parse(self, signature, completion):
        return {"answer": last_boxed(completion) or ""}


def practice_set() -> list[dspy.Example]:
    """The paper's MATH train practice problems (CC BY 4.0, from its released logs)."""
    path = RUN_DIR.parent / "practice_math_train.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PRACTICE_URL, path)
    rows = [json.loads(line) for line in path.open()]
    return [dspy.Example(question=r["problem"], answer=r["answer"]).with_inputs("question") for r in rows]


def math500(n: int) -> list[dspy.Example]:
    from datasets import load_dataset

    rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    random.Random(0).shuffle(rows)
    return [dspy.Example(question=r["problem"], answer=r["answer"]).with_inputs("question") for r in rows[:n]]


def student() -> MLXBackend:
    return MLXBackend(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",  # pin the base so the patch names it exactly
        batch_size=64,
        max_tokens=32_768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        chat_template_kwargs={"enable_thinking": True},
    )


def baseline(n: int) -> None:
    """Score the unedited model on n MATH-500 problems and report how often answers hit the token limit."""
    import time

    from dspy_wsp.forward import ForwardTool

    backend = student()
    program = dspy.Predict(MathQA)
    items = [(f"val_{i}", ex) for i, ex in enumerate(math500(n))]
    tool = ForwardTool(backend=backend, program=program, metric=metric, practice={}, edit_store=None, work_dir=RUN_DIR)
    t = time.time()
    rows = tool.run_items(tool._program(None), items, 1, "original")
    summary = ForwardTool.summarize(rows)
    summary["seconds"] = round(time.time() - t)
    summary["tokens_per_second"] = round(sum(r["output_tokens"] for r in rows) / max(1, summary["seconds"]))
    out = RUN_DIR / f"baseline_{n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    ForwardTool._write_rows(out, rows)
    print(json.dumps(summary, indent=1))
    print(f"rows: {out}")


def main() -> None:
    backend = student()
    optimizer = WeightSpace(
        metric=metric,
        writer_lm=dspy.LM("openai/gpt-6-luna"),
        backend=backend,
        max_passes=MAX_PASSES,
        limits=Limits(max_generations=128, max_generations_both=64),  # smaller passes than the paper's 512 / 256
        task_description="The program solves competition math problems; the answer field holds the final answer in \\boxed{}.",
        sandbox="none",  # bubblewrap is Linux-only; the writer's code runs with your permissions
        run_dir=RUN_DIR,
        resume=True,
    )
    compiled = optimizer.compile(dspy.Predict(MathQA), trainset=practice_set(), valset=math500(VAL_SIZE))
    print(compiled.wsp_patch.summary())
    print(compiled.wsp_report["validation"]["paired"])
    compiled.save(str(RUN_DIR / "program.json"))


if __name__ == "__main__":
    # One LM call per problem: with JSON fallback on, every unparseable 32k-token answer would be regenerated.
    dspy.configure(adapter=BoxedAnswerAdapter(use_json_adapter_fallback=False))
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, metavar="N", help="only score the unedited model on N MATH-500 problems")
    args = parser.parse_args()
    baseline(args.baseline) if args.baseline else main()
