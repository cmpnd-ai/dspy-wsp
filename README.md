# dspy-wsp

Weight-space programming as a DSPy optimizer.

`WeightSpace` improves a DSPy program by having a frontier LM, the *writer*, edit the student model's weights directly. The writer reads the weights in a sandboxed Python kernel. It runs your program on practice examples through a budgeted forward-pass tool, works out what goes wrong inside the network, and saves targeted edits. It uses no gradients and no training.

The method comes from Takehi, Teoh, Zhang and Khattab, *Sparks of Weight Space Programming* (MIT CSAIL, 2026). In their Experiment 8, a frontier model edited 0.016% of Qwen3.5-0.8B's weights. The edit raised MATH-500 from 7.9% to 48.1% by removing a direction that drove the model's looping. This package rebuilds that harness around DSPy programs and metrics.

## Install

```bash
pip install -e ".[hf]"      # from a clone; adds transformers for the Hugging Face backend
pip install -e ".[mlx]"     # or: mlx-lm for the batched backend on Apple silicon
pip install -e ".[vllm]"    # or: vLLM for the batched backend on NVIDIA GPUs
apt install bubblewrap       # sandboxes the writer's code (Linux)
```

## Use

```python
import dspy
from dspy_wsp import WeightSpace, HFBackend

backend = HFBackend("Qwen/Qwen3.5-0.8B", revision="<commit>", max_tokens=8192,
                    chat_template_kwargs={"enable_thinking": True})

def metric(example, pred, trace=None):
    return float(pred.answer.strip() == example.answer)

program = dspy.Predict("question -> answer")

optimizer = WeightSpace(
    metric=metric,
    writer_lm=dspy.LM("openai/gpt-6-luna"),
    backend=backend,
    max_passes=50,
    run_dir="runs/qwen-math",
)
compiled = optimizer.compile(program, trainset=train, valset=val)

compiled(question="What is 17 * 23?")   # runs on the edited weights
print(compiled.wsp_patch.summary())
print(compiled.wsp_report["validation"]["paired"])
```

`trainset` becomes the writer's practice set. The writer never sees `valset`; the optimizer uses it once at the end, comparing original and edited weights on the same examples with the same seeds. Metrics can return a number, a bool, or `dspy.Prediction(score=..., feedback=...)` as GEPA metrics do. The writer reads any feedback text in the generation records.

## Backends

A backend holds the student model, swaps edits in and out, and runs every forward pass.

- **`HFBackend`** runs transformers in-process on CUDA, MPS or CPU. It generates one answer at a time, so it suits small models and short answers.
- **`MLXBackend`** runs mlx-lm on Apple silicon. It decodes up to `batch_size` answers together, and the forward-pass tool runs that many programs at once to fill the batch. Each answer samples from its own seed. It also supports the presence penalty in Qwen's recommended sampling settings. On an M2 Max it generates about 650 tokens per second across a batch of 32 answers from Qwen3.5-0.8B, which is several times faster than `HFBackend` on the same machine.

```python
from dspy_wsp import MLXBackend

backend = MLXBackend("Qwen/Qwen3.5-0.8B", revision="<commit>", batch_size=64, max_tokens=32_768,
                     presence_penalty=1.5, chat_template_kwargs={"enable_thinking": True})
```

`MLXBackend` reads attention probabilities only from softmax-attention blocks. In Qwen3.5 those are blocks 3, 7, 11, 15, 19 and 23; the other blocks use linear attention. `examples/math500_qwen_mlx.py` runs the paper's MATH setup on a Mac. Its `--baseline N` flag scores the unedited model without a writer.

- **`VLLMBackend`** runs vLLM on NVIDIA GPUs for runs at the paper's scale. vLLM schedules every concurrent LM call, so the forward-pass tool keeps a whole pass in flight (`concurrency=512`). Teacher-forced reads go to an `HFBackend` on the same GPU, because vLLM cannot return hidden states or MLP activations. Both models hold the same edits. To swap edits, the backend waits for running requests to finish, reloads the checkpoint in place with the edited tensors substituted, and clears vLLM's prefix cache. `gpu_memory_utilization` (default 0.6) sets vLLM's share of GPU memory; the Hugging Face model uses the rest. `batch_invariant=True` makes each answer depend only on its prompt and seed, at some cost in speed.

```python
from dspy_wsp import VLLMBackend

backend = VLLMBackend("Qwen/Qwen3.5-0.8B", revision="<commit>", max_tokens=32_768, presence_penalty=1.5,
                      chat_template_kwargs={"enable_thinking": True})
```

`examples/math500_qwen_vllm.py` runs the paper's MATH setup on one GPU: 50 passes of up to 512 answers, and a final check on all 500 MATH-500 problems with 4 samples each. `results/2026-11-04-math500-qwen-vllm-1h/` documents a one-hour proof-of-concept run on eight A100 GPUs, with the checks we ran before it.

## Decision models

`WeightSpace` also edits System One decision models: models that read a state and typed questions (Choice, Noul and Score) and return a probability for every option, with no text generation. Two are supported, each with our own loader in MLX and in PyTorch:

- `D1Backend` runs Liquid AI's d1-omni-600M (text path).
- `NanoJevBackend` runs NanoJev (a Qwen3-0.6B backbone with a decision head).

The backend runs a local `/v1/systemone` server, and a DSPy program calls it through DSPy's TypeSafe client, as it would call TypeSafe's hosted Jev. Give the program's signatures `Choice`, `Noul` or `Score` outputs from `dspy.experimental`.

```python
from dspy_wsp import D1Backend, WeightSpace

backend = D1Backend(framework="mlx")
optimizer = WeightSpace(metric=metric, writer_lm=dspy.LM("openai/gpt-6.1-sol"), backend=backend,
                        max_passes=25, holdout=300, holdout_key=lambda ex: ex.workflow)
compiled = optimizer.compile(program, trainset=train, valset=test)
```

For a decision model, the writer gets `decide` passes, which run the program and score it, and `read` passes, which return each option's logit and the decision head's inputs. It may recompute the head from those reads and derive edits in closed form. `holdout` moves training examples to a validation set that the writer can score edits on but never see. `examples/typed_decisions.py` runs the Typed Decisions benchmark. `results/2026-11-04-typed-decisions-d1/` documents a run in which one closed-form edit to d1-omni-600M's scorer raised test accuracy from 53.4% to 60.6%.

## How a run works

The writer works in turns. Each turn it explains its next step and sends exactly one block:

- **` ```python `** runs in a persistent kernel with no GPU and, under the sandbox, no network. The kernel predefines `NAMES`, `SHAPES`, `load(name)`, `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()` and `tok`. Gradients are disabled.
- **` ```forward `** sends a JSON request to the forward-pass tool, the only way to run the model:
  - `generate` runs your program on up to 512 practice items with the original weights, the edited weights, or `"both"`. Every answer is scored with your metric. `"both"` uses the same seed for each item and reports the paired gain with its standard error. The records keep every LM call's full prompt and completion. Pooled activations are optional.
  - `teacher_forced` reads up to 1,024 texts and returns log-probs, top-k predictions, hidden states, MLP activations or attention probabilities as `.npy` files.

Each forward pass costs one unit of budget, whatever its size. A refused request costs nothing. The session continues until the writer has used every pass. Then it leaves its chosen edit in `work/edited/` and writes a final answer.

The harness follows the paper's rules:

- **Guardrails:** `save_edit` refuses edits to the embeddings, the output layer and the final norm. It also refuses edits that only rescale a whole tensor. The host checks every edit again before each pass.
- **Limits:** the writer gets 300 Python runs, 15 minutes per run, and 10,000 characters of output per run.
- **Explanations:** a block sent without an explanation is returned unrun, at most twice in a row.
- **Handoffs:** every 40 turns the conversation restarts with the task, the writer's `notes.md` and its current edit.

Every turn is written to disk. Pass `resume=True` with the same `run_dir` to continue an interrupted run.

## What you get back

```
runs/qwen-math/
  task_prompt.txt      what the writer was told
  turns/tNNN/          each turn's prompt and reply
  transcript.jsonl     every Python run and forward pass
  work/                the writer's folder: notes.md, edited/, gen_k.jsonl, fwd_k/, practice/
  final_answer.md      the writer's account of its edit and the evidence
  final_check/         the valset comparison (never shown to the writer)
  patch/               manifest.json + edit.npz
  report.json
```

`compiled.save("program.json")` records the base model, its settings and the patch location. `program.load("program.json", allow_unsafe_lm_state=True)` rebuilds the edited model; DSPy asks for that flag before it imports a custom engine.

The patch stores only the changed entries, plus the sha256 of every base and edited tensor. Applying it checks both hashes, so anyone with the public base model can rebuild the edited weights bit for bit.

To deploy, write a full checkpoint and serve it like any other model:

```bash
dspy-wsp apply ~/.cache/huggingface/.../Qwen3.5-0.8B runs/qwen-math/patch qwen-edited/
vllm serve qwen-edited/
```

```python
program.set_lm(dspy.LM("openai/qwen-edited", api_base="http://localhost:8000/v1"))
```

## Safety

The writer's code runs on your machine. With `sandbox="auto"` (the default), the kernel runs under bubblewrap: the filesystem is read-only except the work folder, and there is no network. If `bwrap` is missing, the optimizer refuses to start. `sandbox="none"` runs the writer's code with your permissions and network; use it only on a machine you're willing to hand to the writer.

## Differences from the paper

- The task is your DSPy program and metric, not fixed math benchmarks. Generation passes run the program; the records show every LM call.
- The paper's writer was `gpt-6-astra` at maximum reasoning effort, driven through the Codex CLI with every built-in tool switched off. Here the writer is any `dspy.LM`. Nobody knows yet how far other models can reproduce the paper's results.
- `activations` takes separate `layers` (hidden-state indices) and `mlp_layers` (block indices) lists, because one list served both in the paper.
- `fallback_if_worse=True` returns the original weights when the final check shows a loss. This is an addition to the paper's protocol, and it is off by default.
- Patches store changed entries or whole tensors. The paper's release also has a low-rank form, which this version doesn't produce yet.

## Limits and roadmap

- `HFBackend` generates one sequence at a time in-process. That's fine for small models and short answers, but too slow at the paper's scale. There, one pass produced 512 answers of up to 32k tokens, and a pass took up to 30 minutes on a server. `MLXBackend` batches on a Mac, but a 64-problem paired pass at 32k tokens still takes one to two hours. Next on the list is a vLLM backend that pushes edited tensors to a running server.
- A run is long and costly. The paper's 50-pass run took 158 turns and 20 hours. Budget for the writer's tokens accordingly.
- Editing removes obstacles; it doesn't add abilities. The paper's Gemma 3 270M follow-up barely moved. That model didn't loop, so the writer had no failure mechanism to repair.
- Neither the paper nor this package compares against prompt optimization yet. Run GEPA on the same student and budget before you trust a gain.

## Credits and license

Code: MIT. The writer's task prompt is adapted from the Experiment 8 prompt released by Takehi et al. in [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) under CC BY 4.0.
