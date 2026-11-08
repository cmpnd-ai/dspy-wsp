# dspy-wsp

Weight-space programming as a DSPy optimizer.

`WeightSpace` improves a DSPy program by having a frontier LM, the *writer*, edit the student model's weights directly. The writer reads the weights in a sandboxed Python kernel. It runs your program on practice examples through a budgeted forward-pass tool, works out what goes wrong inside the network, and saves targeted edits. It uses no gradients and no training.

The method comes from [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL, 2026). In their Experiment 8, a frontier model edited 0.016% of Qwen3.5-0.8B's weights. The edit raised MATH-500 from 7.9% to 48.1% by removing a direction that drove the model's looping. This package rebuilds that harness around DSPy programs and metrics.

This is an early release. We are releasing it outside the core DSPy library so that more people can try it and tell us what breaks. Please report problems and results in [GitHub issues](https://github.com/cmpnd-ai/dspy-wsp/issues) or on the DSPy Discord.

## Install

```bash
pip install dspy-wsp
apt install bubblewrap       # sandboxes the writer's code (Linux)
```

The base install runs every supported decision model, with MLX on Apple silicon and PyTorch elsewhere. Two extras add more:

- `pip install "dspy-wsp[vllm]"` adds the vLLM backend for text-generating LLMs on NVIDIA GPUs.
- `pip install "dspy-wsp[examples]"` adds the datasets for the benchmark scripts in `examples/`.

## Use

`WeightSpace` works like the other DSPy optimizers. It takes your metric and your module, and `compile` returns the improved module. Here the student is d1-omni-600M, a decision model that runs on a laptop:

```python
from typing import Literal

import dspy
from dspy_wsp import WeightSpace, load_decision_backend


class Triage(dspy.Signature):
    """Triage a support ticket."""

    ticket: str = dspy.InputField()
    urgent: bool = dspy.OutputField(desc="Does this ticket need a reply today?")
    team: Literal["billing", "technical", "account"] = dspy.OutputField(desc="Which team should handle it?")


def metric(example, pred, trace=None):
    return ((pred.urgent == example.urgent) + (pred.team == example.team)) / 2


backend = load_decision_backend("d1")   # d1-omni-600M at the tested revision
optimizer = WeightSpace(
    metric=metric,
    writer_lm=dspy.LM("openai/gpt-6-astra", model_type="responses", reasoning_effort="max"),
    backend=backend,
    max_passes=25,
    holdout=300,
    run_dir="runs/triage",
)
compiled = optimizer.compile(dspy.Predict(Triage), trainset=train, valset=test)

compiled(ticket="I was charged twice this month.")   # runs on the edited weights
print(compiled.wsp_patch.summary())
print(compiled.wsp_report["validation"]["paired"])
```

`trainset` becomes the writer's practice set. `holdout` moves 300 of those examples to a validation set that the writer can score edits on but never see. The writer never sees `valset`; the optimizer uses it once at the end, comparing original and edited weights on the same examples with the same seeds. Metrics can return a number, a bool, or `dspy.Prediction(score=..., feedback=...)` as GEPA metrics do. The writer reads any feedback text in the records.

A decision model reads `bool` outputs as yes or no questions and `Literal` outputs as choices. Each output needs a question in its `desc`. The `Choice`, `Noul` and `Score` types in `dspy.experimental` add criteria for each option and ordered score levels, and they return each option's probability.

## Results

We edited three decision models on the Typed Decisions benchmark with `gpt-6-astra` at maximum reasoning effort as the writer. The writer had one conversation with 25 forward passes for each model. The table scores the answers each program returns on 400 held-out test cases. Each gain is the paired gain over the plain model in points, with its standard error.

| Version | d1-omni-600M | NanoJev (Qwen3-0.6B) | d1-2B |
|---|---|---|---|
| Plain | 50.9% | 39.2% | 60.6% |
| ReAnchor | 60.2% (+9.3 ± 1.1) | 50.7% (+11.6 ± 1.3) | 70.1% (+9.5 ± 1.2) |
| WeightSpace | 66.9% (+16.1 ± 1.2) | 52.4% (+13.3 ± 1.1) | 74.0% (+13.4 ± 1.2) |
| WeightSpace + ReAnchor | 69.1% (+18.3 ± 1.3) | 56.4% (+17.3 ± 1.3) | 77.9% (+17.3 ± 1.3) |

ReAnchor fits how the program turns probabilities into answers and changes no weights. Each run took between 78 and 167 minutes and cost between $5.42 and $7.63 in writer calls. With `gpt-6.1-sol` as the writer, the WeightSpace edits alone gained 5.8, 12.1 and −0.6 points on the same three models.

[The results folder](results/2026-11-07-typed-decisions-astra/) has every run, edit and comparison. [Decision models](#decision-models) covers the earlier runs and the Procedural Typed Decisions benchmark.

## Decision models

`WeightSpace` also edits System One decision models: models that read a state and typed questions (Choice, Noul and Score) and return a probability for every option, with no text generation. Three families are supported, each with our own loader in MLX and in PyTorch. Each family's code is in its own folder under `src/dspy_wsp/decision/families/`:

- `D1Backend` runs Liquid AI's d1-omni models (text path), e.g. d1-omni-600M: an encoder with a decision head.
- `NanoJevBackend` runs NanoJev models: a Qwen3 backbone with a decision head.
- `D1LMBackend` runs Liquid AI's d1 models built on LFM3, e.g. d1-2B. It reads each answer from the language model's output at the question's answer position, so it has no separate head. Its output layer is the token embedding, which stays protected, so the writer's edits land in the backbone. It reads text states and answers Choice, Noul and Score questions. It refuses d1-2B's other question types (text, number, date, box and multiple choice).

We have tested `WeightSpace` on three decision models so far: d1-omni-600M, NanoJev and d1-2B. If you run it on another open decision model, please tell us how it went in [GitHub issues](https://github.com/cmpnd-ai/dspy-wsp/issues) or on the DSPy Discord.

`load_decision_backend` opens a model by alias, Hugging Face repo id or local folder. It reads the checkpoint's config to pick the family, and it refuses other architectures with a message that names the supported ones. The aliases `"d1"`, `"nanojev"` and `"d1-2b"` open the checkpoints we tested, at pinned revisions.

```python
from dspy_wsp import load_decision_backend

backend = load_decision_backend("nanojev")                      # C-Tianyu/NanoJev at the tested revision
backend = load_decision_backend("someone/nanojev-finetune")     # any checkpoint of a supported family
backend = load_decision_backend("/path/to/checkpoint", framework="torch")
```

Before you run a new checkpoint, check it:

```bash
dspy-wsp check someone/nanojev-finetune
```

The check loads the model, confirms that its tensors map to our network, and answers sample Choice, Noul and Score questions. It confirms that the answers are valid distributions and that a request gives the same answer alone and in a batch. Finally, it edits one tensor, confirms that the answers move, and confirms that restoring the tensor restores them. `check_decision_model` runs the same check from Python.

The backend runs a local `/v1/systemone` server, and a DSPy program calls it through DSPy's TypeSafe client, as it would call TypeSafe's hosted Jev. Give the program's signatures `Choice`, `Noul` or `Score` outputs from `dspy.experimental`.

```python
from dspy_wsp import WeightSpace, load_decision_backend

backend = load_decision_backend("d1", framework="mlx")
optimizer = WeightSpace(metric=metric, writer_lm=dspy.LM("openai/gpt-6.1-sol"), backend=backend,
                        max_passes=25, holdout=300, holdout_key=lambda ex: ex.workflow)
compiled = optimizer.compile(program, trainset=train, valset=test)
compiled.wsp_patch.write_checkpoint(backend.weights_dir, "edited-d1/")
```

`write_checkpoint` copies the model's folder, including its subfolders and the model's own code files, with the edited tensors substituted and checked against their hashes. The result has the same layout as the base folder, so the model's own code and `load_decision_backend` both open it. The example scripts take `--model` with an alias, repo id or folder, and they save the edited checkpoint in the run folder's `checkpoint/`.

For a decision model, the writer gets `decide` passes, which run the program and score it, and `read` passes, which return each option's logit and the decision head's inputs. It may recompute the head from those reads and derive edits in closed form. `holdout` moves training examples to a validation set that the writer can score edits on but never see. `examples/typed_decisions.py` runs the Typed Decisions benchmark. Two runs on a MacBook Pro with MLX, with `openai/gpt-6.1-sol` as the writer and 25 passes each, gave these results on the 400 held-out test cases:

| Model | Tensor edited | Original | Edited | Paired gain | Improved / worse | Run |
|---|---|---|---|---|---|---|
| d1-omni-600M | `head.scorer.3.weight` (1,024 weights) | 53.4% | 60.6% | +7.25 ± 1.05 points | 150 / 63 | [results](results/2026-11-04-typed-decisions-d1/) |
| NanoJev (Qwen3-0.6B) | `scalar.weight` (1,024 weights) | 38.6% | 55.7% | +17.10 ± 1.35 points | 243 / 63 | [results](results/2026-11-04-typed-decisions-nanojev/) |

In both runs the writer derived the edit in closed form from the decision head's inputs, confirmed it on 300 held-out validation cases, and never saw the test cases. Each run took between 75 and 110 minutes.

WeightSpace also combines with `dspy.experimental.ReAnchor`, which fits how a program turns probabilities into answers (Noul thresholds, Score cut points and Choice weights) without changing weights. Scored on the answers NanoJev's program returns, on the same 400 test cases:

| NanoJev | Decided accuracy | Paired gain over plain |
|---|---|---|
| Plain | 39.2% | |
| ReAnchor | 50.7% | +11.6 ± 1.3 points |
| WeightSpace | 51.3% | +12.1 ± 1.2 points |
| WeightSpace + ReAnchor | 55.7% | +16.6 ± 1.4 points |

See [the comparison](results/2026-11-04-typed-decisions-nanojev/reanchor/) for how the two methods divide the gain.

On three balanced, rule-based tasks from Procedural Typed Decisions, overnight on the same laptop, the same kind of edit gave larger gains on 400 test cases:

| Model | Tensor edited | Original | Edited | Paired gain | Improved / worse | Run |
|---|---|---|---|---|---|---|
| d1-omni-600M | `head.scorer.3.weight` (1,024 weights) | 52.3% | 72.3% | +20.0 ± 2.3 points | 179 / 54 | [results](results/2026-11-04-procedural-decisions/) |
| NanoJev (Qwen3-0.6B) | `scalar.weight` (1,024 weights) | 49.3% | 72.3% | +23.1 ± 2.3 points | 193 / 54 | [results](results/2026-11-04-procedural-decisions/) |

These tasks have balanced labels. Edits that only shifted option preferences gained nothing, so the gains come from how the head reads each record.

Scored on the answers each program returns, on the same 400 test cases:

| Version | d1-omni-600M | NanoJev |
|---|---|---|
| Plain | 52.3% | 47.8% |
| ReAnchor | 58.7% (+6.4 ± 2.2) | 52.8% (+5.0 ± 1.7) |
| WeightSpace | 64.2% (+11.9 ± 2.3) | 68.4% (+20.6 ± 2.3) |
| WeightSpace + ReAnchor | 69.9% (+17.7 ± 2.4) | 71.3% (+23.4 ± 2.3) |
| WeightSpace (GEPA) | 66.9% (+14.7 ± 2.2) | 65.6% (+17.8 ± 2.4) |
| WeightSpace (GEPA) + ReAnchor | 71.2% (+18.9 ± 2.3) | 67.9% (+20.1 ± 2.4) |

The WeightSpace rows come from one 25-pass conversation. The GEPA rows come from a [population search](#population-search) with 51 passes and at least 10 rounds. The search beat the conversation on d1-omni-600M and trailed it on NanoJev. Every difference between the two is smaller than two standard errors.

## Generating LLMs

`WeightSpace` also edits text-generating LLMs, as the paper did. A pass on an LLM generates many long answers, so we recommend `VLLMBackend` on a GPU machine. `HFBackend` and `MLXBackend` suit small experiments.

A backend holds the student model, swaps edits in and out, and runs every forward pass.

- **`VLLMBackend`** runs vLLM on NVIDIA GPUs for runs at the paper's scale. vLLM schedules every concurrent LM call, so the forward-pass tool keeps a whole pass in flight (`concurrency=512`). Teacher-forced reads go to an `HFBackend` on the same GPU, because vLLM cannot return hidden states or MLP activations. Both models hold the same edits. To swap edits, the backend waits for running requests to finish, reloads the checkpoint in place with the edited tensors substituted, and clears vLLM's prefix cache. `gpu_memory_utilization` (default 0.6) sets vLLM's share of GPU memory; the Hugging Face model uses the rest. `batch_invariant=True` makes each answer depend only on its prompt and seed, at some cost in speed.

```python
from dspy_wsp import VLLMBackend

backend = VLLMBackend("Qwen/Qwen3.5-0.8B", revision="<commit>", max_tokens=32_768, presence_penalty=1.5,
                      chat_template_kwargs={"enable_thinking": True})
```

- **`HFBackend`** runs transformers in-process on CUDA, MPS or CPU. It generates one answer at a time, so it suits small models and short answers.
- **`MLXBackend`** runs mlx-lm on Apple silicon. It decodes up to `batch_size` answers together, and the forward-pass tool runs that many programs at once to fill the batch. Each answer samples from its own seed. It also supports the presence penalty in Qwen's recommended sampling settings. On an M2 Max it generates about 650 tokens per second across a batch of 32 answers from Qwen3.5-0.8B, which is several times faster than `HFBackend` on the same machine.

```python
from dspy_wsp import MLXBackend

backend = MLXBackend("Qwen/Qwen3.5-0.8B", revision="<commit>", batch_size=64, max_tokens=32_768,
                     presence_penalty=1.5, chat_template_kwargs={"enable_thinking": True})
```

`MLXBackend` reads attention probabilities only from softmax-attention blocks. In Qwen3.5 those are blocks 3, 7, 11, 15, 19 and 23; the other blocks use linear attention.

## Replicate the paper

The paper's Experiment 8 edited Qwen3.5-0.8B to stop its looping on MATH-500. Two scripts rebuild that setup:

- `examples/math500_qwen_vllm.py` runs it on one GPU: 50 passes of up to 512 answers, and a final check on all 500 MATH-500 problems with 4 samples each.
- `examples/math500_qwen_mlx.py` runs a smaller version on a Mac. Its `--baseline N` flag scores the unedited model without a writer.

This path is a bigger step than the decision models, and we have tested it less. A full run at the paper's scale takes many GPU hours and a long writer session. `results/2026-11-04-math500-qwen-vllm-1h/` documents a one-hour proof-of-concept run on eight A100 GPUs, with the checks we ran before it. We have not yet repeated the paper's full result. If you run this setup, please share what you find in [GitHub issues](https://github.com/cmpnd-ai/dspy-wsp/issues) or on the DSPy Discord.

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

### Population search

By default, a run is one long conversation. For decision models, `search=GEPASearch(...)` runs GEPA's population search instead. The writer still works in the Python kernel with the same tools. The search keeps a population of candidate edits and runs in rounds:

1. GEPA picks a parent from its Pareto frontier over the validation examples.
2. GEPA scores the parent on a minibatch of practice examples.
3. The writer gets a short session that starts from the parent's edit and notes, with the minibatch's feedback and a small budget of forward passes. The edit it leaves becomes the child.
4. GEPA scores the child on the same minibatch. If the child scores higher, GEPA scores it on the validation set and adds it to the population.

```python
from dspy_wsp import GEPASearch, WeightSpace

optimizer = WeightSpace(metric=metric, writer_lm=writer, backend=backend, max_passes=51, holdout=300,
                        search=GEPASearch(min_rounds=10, minibatch_size=8, passes_per_proposal=2))
```

- `max_passes` counts every pass: the writer's passes and each of GEPA's scorings. A round costs at most `3 + passes_per_proposal` passes.
- `WeightSpace` refuses a `max_passes` that can't pay for `min_rounds` rounds, and a `max_code_runs` below `min_rounds * code_runs_per_proposal` (30 by default).
- A parent that scores perfectly on its minibatch can't be beaten there, so GEPA skips that round's writer session. The parent's scoring still costs one pass.
- A writer call that times out, drops its connection or runs past the model's output limit is retried twice. If it still fails, or every example fails to run, the search stops. The run then finishes with the best candidate so far, and the report's `search.error` says why.
- Writers can't run validation passes in a round, because GEPA uses the validation set to select candidates.
- Each round is written to `rounds.jsonl` and `progress.md` when it ends. Every candidate's edit and notes are kept in `candidates/`.
- Pass `resume=True` with the same `run_dir` to continue a stopped search. It picks up GEPA's saved state, the round log and the pass count, and it reuses the seed's saved validation scores.

The example scripts take `--search gepa --rounds 10`. To compare the two searches, give the conversation the same `--passes`.

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
- The paper's writer was `gpt-6-astra` at maximum reasoning effort, driven through the Codex CLI with every built-in tool switched off. Here the writer is any `dspy.LM`. The writer matters: on our decision model runs, `gpt-6-astra` wrote better edits than `gpt-6.1-sol` on two of three models.
- `activations` takes separate `layers` (hidden-state indices) and `mlp_layers` (block indices) lists, because one list served both in the paper.
- `fallback_if_worse=True` returns the original weights when the final check shows a loss. This is an addition to the paper's protocol, and it is off by default.
- Patches store changed entries or whole tensors. The paper's release also has a low-rank form, which this version doesn't produce yet.

## Limits and roadmap

- `HFBackend` generates one sequence at a time in-process. That's fine for small models and short answers, but too slow at the paper's scale. There, one pass produced 512 answers of up to 32k tokens, and a pass took up to 30 minutes on a server. `MLXBackend` batches on a Mac, but a 64-problem paired pass at 32k tokens still takes one to two hours.
- A run is long and costly. The paper's 50-pass run took 158 turns and 20 hours. Budget for the writer's tokens accordingly.
- Editing removes obstacles; it doesn't add abilities. The paper's Gemma 3 270M follow-up barely moved. That model didn't loop, so the writer had no failure mechanism to repair.
- Neither the paper nor this package compares against prompt optimization yet. Run GEPA on the same student and budget before you trust a gain.

## Credits and license

Code: MIT.

- The method and the harness come from [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. Please cite it when you use this package.
- The writer's task prompt is adapted from the Experiment 8 prompt released by Takehi et al. in [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) under CC BY 4.0.
- Haoze "Hector" He's [weight editing design](https://github.com/HectorHHZ/llm-honing/blob/main/system-design/weight-edit.md) for `llm-honing` is earlier work on editing decision models, including NanoJev, and releasing the edits as checked checkpoints. See [his post](https://x.com/HaozeHe_Hector/status/2118048048148623453).
- Michael Isaac contributed the GEPA population search. It uses Lakshya Agrawal's [GEPA](https://github.com/gepa-ai/gepa) library.

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```
