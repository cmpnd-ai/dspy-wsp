"""Edit Qwen3.5-0.8B's weights to improve MATH-500, at the paper's scale on an NVIDIA GPU.

This is the paper's Experiment 8 setup: thinking on, the model card's sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 tokens),
50 passes of up to 512 answers, and a final check on all 500 MATH-500 problems with
4 samples each. The writer practices on the paper's 7,275 MATH train problems.

    pip install -e ".[examples,vllm]"
    apt install bubblewrap
    python examples/math500_qwen_vllm.py --baseline 500   # the unedited model only; no writer, no API cost
    python examples/math500_qwen_vllm.py --dp 8 --gpu-mem 0.8   # 8 GPUs, one model copy each
    python examples/math500_qwen_vllm.py --passes 3 --pass-size 128 --val-size 32 --val-samples 1 --run-dir runs/short   # a short end-to-end check
    OPENAI_API_KEY=... python examples/math500_qwen_vllm.py

One H100 runs a pass in roughly 10 to 15 minutes. Rerun the script to resume.
"""

import argparse
import json
import time
from pathlib import Path

import dspy
from math500_qwen_mlx import BoxedAnswerAdapter, MathQA, PaperPromptAdapter, math500, metric, practice_set

from dspy_wsp import Limits, VLLMBackend, WeightSpace

RUN_DIR = Path("runs/math500-qwen-vllm")
MAX_PASSES = 50
VAL_SAMPLES = 4


def student(gpu_memory_utilization: float = 0.6, data_parallel_size: int = 1, split_thinking: bool = True) -> VLLMBackend:
    """gpu_memory_utilization: vLLM's share of each GPU's memory; the Hugging Face model for reads uses the rest of GPU 0.
    data_parallel_size: GPUs that each hold a full copy of the model and serve part of the batch.
    split_thinking: give the adapter only the text after </think> (False passes the whole reply).
    """
    return VLLMBackend(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",  # pin the base so the patch names it exactly
        concurrency=512,
        max_tokens=32_768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        chat_template_kwargs={"enable_thinking": True},
        gpu_memory_utilization=gpu_memory_utilization,
        split_thinking=split_thinking,
        engine_kwargs={
            "limit_mm_per_prompt": {"image": 0, "video": 0},  # text only: skip the vision encoder's memory
            "data_parallel_size": data_parallel_size,
        },
    )


def baseline(n: int, gpu_mem: float, dp: int, paper_prompt: bool) -> None:
    """Score the unedited model on n MATH-500 problems and report speed and cut-off answers."""
    from dspy_wsp.forward import ForwardTool

    backend = student(gpu_mem, dp, split_thinking=not paper_prompt)
    items = [(f"val_{i}", ex) for i, ex in enumerate(math500(n))]
    tool = ForwardTool(backend=backend, program=dspy.Predict(MathQA), metric=metric, practice={}, edit_store=None, work_dir=RUN_DIR)
    t = time.time()
    rows = tool.run_items(tool._program(None), items, 1, "original")
    summary = ForwardTool.summarize(rows)
    summary["seconds"] = round(time.time() - t)
    summary["tokens_per_second"] = round(sum(r["output_tokens"] for r in rows) / max(1, summary["seconds"]))
    out = RUN_DIR / f"baseline_{n}{'_paper_prompt' if paper_prompt else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    ForwardTool._write_rows(out, rows)
    print(json.dumps(summary, indent=1))
    print(f"rows: {out}")


def main(args) -> None:
    run_dir = Path(args.run_dir)
    limits = Limits()  # the paper's pass sizes: 512 answers, or 256 per model for "both"
    if args.pass_size:
        limits = Limits(max_generations=args.pass_size, max_generations_both=args.pass_size // 2)
    optimizer = WeightSpace(
        metric=metric,
        writer_lm=dspy.LM("openai/gpt-6-luna"),
        backend=student(args.gpu_mem, args.dp, split_thinking=not args.paper_prompt),
        max_passes=args.passes,
        limits=limits,
        task_description="The program solves competition math problems; the answer field holds the final answer in \\boxed{}.",
        val_samples=args.val_samples,
        run_dir=run_dir,
        resume=True,
    )
    compiled = optimizer.compile(dspy.Predict(MathQA), trainset=practice_set(), valset=math500(args.val_size))
    print(compiled.wsp_patch.summary())
    print(compiled.wsp_report["validation"]["paired"])
    compiled.save(str(run_dir / "program.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, metavar="N", help="only score the unedited model on N MATH-500 problems")
    parser.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM's share of each GPU's memory")
    parser.add_argument("--dp", type=int, default=1, help="model copies, one per GPU (vLLM data parallelism)")
    parser.add_argument("--passes", type=int, default=MAX_PASSES, help="forward passes the writer may use")
    parser.add_argument("--pass-size", type=int, help="answers per pass (default: the paper's 512, or 256 per model for 'both')")
    parser.add_argument("--val-size", type=int, default=500, help="MATH-500 problems in the final check")
    parser.add_argument("--val-samples", type=int, default=VAL_SAMPLES, help="samples per problem in the final check")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--paper-prompt", action="store_true", help="the paper's bare prompt and last-\\boxed{} grading instead of DSPy's ChatAdapter")
    args = parser.parse_args()
    # One LM call per problem: with JSON fallback on, every unparseable 32k-token answer would be regenerated.
    dspy.configure(adapter=PaperPromptAdapter() if args.paper_prompt else BoxedAnswerAdapter(use_json_adapter_fallback=False))
    baseline(args.baseline, args.gpu_mem, args.dp, args.paper_prompt) if args.baseline else main(args)
