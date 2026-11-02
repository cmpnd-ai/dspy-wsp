"""The writer's task prompt.

Adapted from the Experiment 8 prompt in Takehi, Teoh, Zhang and Khattab, "Sparks
of Weight Space Programming" (2026), released under CC BY 4.0 in
github.com/RikiyaT/lm_understand-logs. Changes: the task is a DSPy program and
its metric instead of fixed math benchmarks; the forward tool runs the program;
paths, limits and the model description come from the run.
"""

from __future__ import annotations

import inspect
import re
from string import Template

_BLOCK = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")

TASK = Template(
    """TASK. You are given the weights of a language model: $model_description It runs every language-model call of the DSPy program described below. Your job is to make the program score higher on its metric by EDITING THE MODEL'S WEIGHTS DIRECTLY: find out what goes wrong inside the network when the program fails, and repair that with targeted changes.$task_description

THE PROGRAM.
$program_description

HOW IT WILL BE MEASURED. After you finish, the program is run with your edited weights and with the original weights on held-out examples that you never see, and every answer is scored with this metric:
$metric_description
Only weight values can change (same tensor names and shapes): the program, its prompts, the tokenizer and the decoding settings stay fixed. Decoding: $sampling.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when the program fails, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts. Out of scope, and refused by save_edit: $protected_description, and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for.

WHAT YOU HAVE.
1. A persistent Python kernel ($kernel_environment): variables survive between your code blocks, like a notebook. The model's files are in $model_dir (read only), practice examples are in $practice_dir, and $work_dir is your read-write folder. Available: numpy, PyTorch (CPU) and the other packages installed with it. Defined in advance: NAMES (all tensor names), SHAPES, load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (the tokenizer). Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One execution may take up to $timeout seconds; output is cut at $max_output characters. You may run code up to $max_code_runs times. If the kernel is restarted (time or memory limit), its variables are lost and you are told so.
Tensor names, with block 0 standing for every block:
$tensor_listing

2. Practice examples with their labels (none is in the evaluation), one JSON object per line with "id", "inputs" and "labels": $practice_file ($n_practice examples). Input fields: $input_keys. Label fields: $label_keys. Choose whichever serve your experiments.

3. $forward_section

HOW TO WORK.
1. Diagnose first. Look at how the program fails on practice examples: a wrong answer, an answer the program could not parse, or an LM call cut off at the token limit (often while the model repeats itself). The generation summary counts these; read some failing records, including the full prompt and completion of each LM call. Go after the largest category you can explain.
2. Localize causally. That a component is more active on good answers is not enough (good answers come from easier and shorter examples). Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. With a 0/1 metric, a score over n generations has a standard error of about 50/sqrt(n) percentage points: about 4.4 points at 128 generations, 2.2 at 512. Compare original and edited weights on the SAME examples: a generation pass with "model": "both" does this with the same random seeds and reports the paired difference with its standard error, which is usually much smaller than that of two separate measurements. Separate passes do not reproduce each other exactly, so never compare scores across passes and never repeat an identical request with unchanged edits. Keep an edit only if its gain is clearly larger than its standard error on examples that were not used to choose it. Dense measurements are cheap: one teacher-forced pass can read many texts and gives a signal on every token. Edits that change a tensor by a fraction of a percent usually have effects too small to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes: for example one diagnostic generation with activations, teacher-forced passes for dense paired measurements, and a final paired generation ("model": "both") on examples you have not used, to confirm.
5. Work efficiently: state the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays (print short excerpts of generated texts, at most about 300 characters each, and token ids rather than raw vocabulary dumps); keep $work_dir/notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit; it is shown to you again if the conversation is restarted); before every forward pass, write the exact current edit and the examples used to notes.md; keep copies of good edits (for example copy $work_dir/edited to $work_dir/best_1/).$explain

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) internet access or search. Allowed: any analysis of the weights and of forward-pass results, any use of the practice examples, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. $ending"""
)

FORWARD = Template(
    """A forward-pass tool with a budget of $max_passes forward passes in total. A forward pass is one of two things, and each counts as one pass whatever its size:
 (a) GENERATION: the program runs on up to $max_gen items (examples x samples) with the evaluation's settings, and every answer to a practice example is scored with the metric. Reply with exactly one block that starts with ```forward and contains JSON:
 {"type": "generate", "model": "original", "edited" or "both", "problems": ["<practice id>", ...], "inputs": [{<the input fields>}, ...] (optional: your own inputs, not scored), "n": samples per item (items x n <= $max_gen; with "both" <= $max_both per model), "max_tokens": per LM call (optional), "activations": {"layers": [hidden-state indices 0..$n_layers], "mlp_layers": [block indices 0..$last_block]} (optional)}
 "original" runs the unedited weights, "edited" your current edits, and "both" runs both on the same items and returns a paired comparison (the score difference edited - original, averaged per item, with its standard error over items, and how many samples improved or worsened). The sampling seed of every item is fixed, in every pass and for both models. You get the mean score, the mean output length, the share of items with an LM call cut at max_tokens and the failure counts; per-item records (inputs, outputs, score, the metric's feedback if any, and every LM call's full prompt and completion) are in $work_dir/gen_<k>.jsonl (with "both": gen_<k>_original.jsonl and gen_<k>_edited.jsonl). With "activations", for every LM call the residual stream at the requested indices (0 = embeddings, l = after block l-1) and the inputs of the listed blocks' MLP output projections are saved, taken at the last prompt token and averaged over the call's first $pooled completion tokens, with each record's score: $work_dir/gen_<k>_acts/ (index.json lists the rows; with "both" one folder per model).
 (b) TEACHER-FORCED: the model reads up to $max_texts texts of up to $max_text_tokens tokens each, at most $max_total_tokens tokens in total (no generation). Reply with a ```forward block containing JSON:
 {"type": "teacher_forced", "model": "original" or "edited", "texts": ["raw text", ...] or "chats": [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...], "return": {"token_logprobs": true, "topk": k (at most $max_topk), "hidden_states": [indices 0..$n_layers], "mlp_act": [block indices], "attn_layers": [block indices; at most $max_att_texts texts of at most $max_att_tokens tokens]}}
 The results are .npy files (with meta.json) in $work_dir/fwd_<k>/, right-padded to the longest text. To score the program's own answers token by token, take an LM call's "prompt" and "completion" from a gen file and send prompt + completion as one raw text. A request that fails does not count against the budget."""
)

NO_FORWARD = "No forward passes: you cannot run the model. Work from the weights alone."

EXPLAIN = "\n6. EXPLAIN EVERY STEP. Before each ```python or ```forward block, explain in plain English what it does and why. A block without an explanation is sent back unrun."


def _source(obj, limit: int = 4000) -> str | None:
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return src if len(src) <= limit else src[:limit] + "\n# ... (cut)"


def describe_program(program) -> str:
    preds = list(program.named_predictors())
    lines = [f"{type(program).__name__}, with {len(preds)} LM-calling step(s); DSPy formats each step's prompt and parses its answer."]
    for name, pred in preds:
        sig = pred.signature
        ins = ", ".join(sig.input_fields)
        outs = ", ".join(sig.output_fields)
        lines.append(f"- step {name!r}: {ins} -> {outs}")
        if sig.instructions:
            lines.append(f"  instructions: {sig.instructions.strip()[:1500]}")
        for fname, f in {**sig.input_fields, **sig.output_fields}.items():
            desc = (f.json_schema_extra or {}).get("desc")
            if desc and desc != f"${{{fname}}}":
                lines.append(f"  {fname}: {desc}")
        if getattr(pred, "demos", None):
            lines.append(f"  {len(pred.demos)} few-shot demonstrations are included in its prompt")
    if not type(program).__module__.startswith("dspy."):
        src = _source(type(program))
        if src:
            lines.append("Program source:\n" + src)
    return "\n".join(lines)


def describe_metric(metric) -> str:
    name = getattr(metric, "__name__", type(metric).__name__)
    src = _source(metric)
    return f"{name}\n{src}" if src else name


def tensor_listing(shapes: dict[str, tuple], dtypes: dict[str, str] | None = None, protected=lambda n: False) -> str:
    dtypes = dtypes or {}
    blocks = sorted({int(m.group(1)) for n in shapes if (m := _BLOCK.search(n))})
    lines = []
    for n in sorted(shapes):
        m = _BLOCK.search(n)
        if m and int(m.group(1)) != (blocks[0] if blocks else 0):
            continue
        tag = " (out of scope)" if protected(n) else ""
        lines.append(f"  {n} {list(shapes[n])}{(' ' + dtypes[n]) if n in dtypes else ''}{tag}")
    if blocks:
        lines.append(f"  ... the same names for blocks {blocks[0]}..{blocks[-1]}; {len(shapes)} tensors in total")
    return "\n".join(lines)


def build_task_prompt(
    *,
    model_description: str,
    task_description: str | None,
    program_description: str,
    metric_description: str,
    sampling: str,
    protected_description: str,
    kernel_environment: str,
    model_dir: str,
    practice_dir: str,
    practice_file: str,
    work_dir: str,
    timeout: int,
    max_output: int,
    max_code_runs: int,
    tensor_listing: str,
    n_practice: int,
    input_keys: list[str],
    label_keys: list[str],
    max_passes: int,
    limits,
    n_layers: int | None,
    require_explanations: bool,
) -> str:
    if max_passes > 0:
        forward = FORWARD.substitute(
            max_passes=max_passes,
            max_gen=limits.max_generations,
            max_both=limits.max_generations_both,
            n_layers=n_layers if n_layers is not None else "L",
            last_block=(n_layers - 1) if n_layers is not None else "L-1",
            pooled=f"{limits.pooled_answer_tokens:,}",
            work_dir=work_dir,
            max_texts=f"{limits.max_texts:,}",
            max_text_tokens=f"{limits.max_text_tokens:,}",
            max_total_tokens=f"{limits.max_total_tokens:,}",
            max_topk=limits.max_topk,
            max_att_texts=limits.max_attention_texts,
            max_att_tokens=f"{limits.max_attention_tokens:,}",
        )
        ending = (
            f"It continues until all {max_passes} forward passes are used (replying without a code block does not end it earlier); "
            f"after that, make your final choice: the tensors in {work_dir}/edited when you finish are submitted, so leave there the edit "
            "with the best measured paired gain (on examples not used to choose it), even if the gain is not statistically significant. "
            "Then reply without a code block, saying what you changed, the evidence for it, its measured gain with its standard error, and why you expect it to help."
        )
    else:
        forward = NO_FORWARD
        ending = (
            f"When you are done, the tensors in {work_dir}/edited are submitted. Reply without a code block, "
            "saying what you changed and why you expect it to help."
        )
    return TASK.substitute(
        model_description=model_description,
        task_description=f" {task_description.strip()}" if task_description else "",
        program_description=program_description,
        metric_description=metric_description,
        sampling=sampling,
        protected_description=protected_description,
        kernel_environment=kernel_environment,
        model_dir=model_dir,
        practice_dir=practice_dir,
        work_dir=work_dir,
        timeout=timeout,
        max_output=f"{max_output:,}",
        max_code_runs=max_code_runs,
        tensor_listing=tensor_listing,
        practice_file=practice_file,
        n_practice=f"{n_practice:,}",
        input_keys=", ".join(input_keys),
        label_keys=", ".join(label_keys) or "(none)",
        forward_section=forward,
        explain=EXPLAIN if require_explanations else "",
        ending=ending,
    )


def handoff(task_prompt: str, notes: str, edit_summary: str, budget_status: str, kernel_restarted: bool) -> str:
    why = (
        "The run was resumed after an interruption: the kernel was restarted and its variables are lost."
        if kernel_restarted
        else "The conversation was restarted to keep it short; the kernel and its variables are unchanged."
    )
    return (
        f"HANDOFF. {why} Your task follows, unchanged.\n\n{task_prompt}\n\n"
        f"YOUR NOTEBOOK (notes.md):\n{notes or '(empty)'}\n\nYOUR CURRENT EDIT:\n{edit_summary}\n\n{budget_status}"
    )
