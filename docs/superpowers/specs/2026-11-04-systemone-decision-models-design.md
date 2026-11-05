# Design: System One decision models in WeightSpace

## Intent

`WeightSpace` lets a frontier writer model read and edit a small model's weights to improve a DSPy program, with no gradients. It works today for language models that generate text. We validated that pipeline on GPUs with a MATH-500 proof of concept.

This project extends `WeightSpace` to decision models. A decision model reads a state and a set of typed questions and returns a probability for every option of every question. It does not generate text. TypeSafe calls this interface System One, and its three question types are Choice, Noul (yes or no) and Score (ordered levels). DSPy 3.4 calls such models through `dspy.experimental.TypeSafe`.

Decision models are a more practical target than long-thinking language models. A decision is one forward pass, so a full practice pass costs minutes, not hours. Applications use them as typed components, e.g., a router or a reviewer. Honing's report on NanoJev in Tank War showed that one direction added to a decision head's weights can repair a failure that fine-tuning did not.

The end state is one optimizer that takes any DSPy module and any metric function, and runs on a language model or a decision model, with PyTorch on CUDA or with MLX on Apple silicon.

## Success

The first success is a run on the Typed Decisions benchmark in which `gpt-6.1-sol`, as the writer, edits a decision model's weights and the test score goes up. The test score is measured on the held-out test split, as a paired comparison of the original and edited weights on the same cases. The gain must be larger than twice its standard error for at least one of the two models.

## Goals

- Support two decision models with our own loaders:
  - NanoJev (`C-Tianyu/NanoJev`, revision `047b927b30882a1138fc504821b82ac145a4b81a`). It is a Qwen3-0.6B backbone with a decision head. Each option is scored on its own text path, and the head turns each path's final hidden state into a logit. The source is `TianyuCodings/NanoJev` (MIT).
  - Liquid AI's d1-omni-600M (`LiquidAI/d1-omni-600M`). It is an LFM2.5-Encoder-350M trunk with a two-layer decision head and calibration temperatures per question type. It also has audio and vision encoders. We support its text path only.
- Run each model with PyTorch on CUDA and with MLX on Apple silicon. Both implementations must give the same answers within a small tolerance.
- Let a DSPy program call each model through DSPy's TypeSafe client, as it would call TypeSafe's hosted Jev.
- Give the writer passes that fit decision models: running the program on practice items, and reading what the model computes for given requests.
- Run `WeightSpace` on Typed Decisions with both models, and document the runs in `results/` the way the MATH-500 run is documented.
- After that, run NanoJev on Tank War and compare with Honing's published result.

## Non-goals

- Changing the language-model pipeline. `HFBackend`, `MLXBackend`, `VLLMBackend`, the forward tool's `generate` and `teacher_forced` passes, and the writer prompt for language models stay as they are. Their tests must pass without changes. New code may be added beside them, and shared code may gain new options, but the default behavior of every existing path must not change.
- Images, audio and video. The decision models read text and JSON state only.
- Language models used as decision models through answer-token probabilities. LFM2.5 was considered and dropped.
- Clef-Flash (9B) and other larger decision models, for now. The design should not rule them out.
- Training of any kind. Edits are computed directly, as in the paper.

## Anti-goals

These would make the project a failure even if it ran:

- Using the test split in any way during optimization, including for choosing the final edit.
- Depending at run time on remote model code (`trust_remote_code`), on Honing's model code, or on the models' own repositories for loading or inference. We may run their reference code in tests, to check that our loaders give the same outputs. Honing's Tank War game (`match.mjs` and its bridge, Apache-2.0) is the environment for the second task, and running it is allowed.
- A baseline that does not match the reference implementation of each model. If our loader disagrees with the reference code, every result built on it is suspect.
- Claiming that our scores compare with the Typed Decisions leaderboard when our prompt or request format differs from the leaderboard's.
- Publishing edited d1 weights before checking the LFM Open License.

## Constraints

- The development machine is an M2 Max with 64 GB of memory. The first runs happen there with MLX. The CUDA path must also work, and we will test it on a rented GPU.
- The writer is `openai/gpt-6.1-sol`, through `dspy.LM`, with no `max_tokens` and no `temperature` set. The user chose this model for these runs. It replaces the usual default, `openai/gpt-6-luna`, in the decision-model examples only.
- DSPy's TypeSafe client imports `typesafe_sdk`, which is not installed yet. The project adds the `dspy[typesafe]` extra and pins the SDK version, because the SDK's wire format defines what our `/v1/systemone` server must accept.
- Licenses:
  - NanoJev's source is MIT. Its model card states no license.
  - d1-omni-600M uses the LFM Open License v1.0, which limits commercial use for organizations with $10 million or more in annual revenue.
  - Typed Decisions is Apache-2.0.
- The kernel sandbox, the edit guardrails, the patch format and the run folder layout stay as they are. Decision models use them unchanged, except for per-model lists of protected tensors.

## How it works

### Decision backends

A new `DecisionBackend` extends `WeightBackend`. It keeps the parts every backend has: the weights folder, `set_weights` and `using` for edits, the seed and call recording, and `editable_names`. It replaces generation with System One answering:

- `system_one(state, questions)` answers one request.
- `system_one_batch(requests)` answers many requests.

A request follows TypeSafe's API. `state` is a string or JSON. `questions` maps names to Choice, Noul or Score questions with `instructions` and `criteria`. Each answer has the fields TypeSafe returns:
- A Choice answer has `choice`, `probabilities` and `confidence`.
- A Score answer has `score`, `probabilities` and `confidence`.
- A Noul answer has `noul`, the probability of yes.

Confidence uses TypeSafe's published formulas. For a Choice with n options and largest probability p, confidence is (n × p − 1) / (n − 1).

There are four implementations: NanoJev and d1, each in PyTorch and in MLX. Each one loads the checkpoint's safetensors directly and builds the network in our code. The two frameworks share the request encoding (tokenizing the state, the questions and the options into the model's input layout), so they differ only in the network code.

How each model handles question types:

- d1 answers Choice, Noul and Score natively, and applies its calibration temperatures.
- NanoJev answers Choice only. The backend sends a Noul as a Choice between yes and no. It sends a Score as a Choice over the levels, and computes the score from the level probabilities.
- NanoJev's state has four sections (`essential`, `details`, `memory` and `logs`). A System One state goes into `essential`. Tank War fills all four sections itself.

Each implementation needs a parity test against the reference code. On a fixed set of requests, including some from Typed Decisions, our probabilities must match the reference within a tolerance the builder chooses and records. The PyTorch and MLX versions must also match each other.

### Serving to DSPy

The backend starts a local HTTP server that answers `POST /v1/systemone` with TypeSafe's request and response bodies. The backend's `lm()` returns a small subclass of `dspy.experimental.TypeSafe` pointed at that server, with DSPy's cache off. A DSPy program with `Choice`, `Noul` or `Score` output fields then calls the model through DSPy's own TypeSafe connector and the SDK's real HTTP calls. The same server works for TypeSafe's SDKs and any other client.

The subclass adds one thing to DSPy's client. Its saved state records the backend's settings and the patch location, as `BackendEngine.dump_state` does for language models. Loading a compiled program rebuilds the backend, applies the patch and starts a new local server. Without this, a saved program would point at a local address that no longer exists.

The server always answers with the backend's current weights. A weight swap waits until no request is running, as the MLX and vLLM backends do today.

For Tank War, the same server also answers Honing's `POST /v1/decide` and `GET /v1/health`, so the game's `match.mjs` can play against the current weights.

### Recording calls

The forward tool's records keep every model call. For decision models, a call is a System One request and its answers: the state, the questions, every option's probability and the chosen answer. These records replace the prompt and completion that language-model records hold. The writer reads them to find failures.

### Writer passes for decision models

When the backend is a decision backend, the forward tool offers two passes:

- `decide` runs the DSPy program on up to the pass limit of practice items with the original weights, the edited weights or both, and scores every item with the metric. It works the same way as `generate`, with the same pairing, records and summaries. Decision models are deterministic, so a pass with both models shows the exact effect of an edit on each item.
- `read` runs given System One requests through the model and returns, for each question and option:
  - the head's inputs,
  - the option logits,
  - the probabilities,
  - and, on request, hidden states at chosen layers and MLP inputs, pooled over the state and kept for each option.

  The writer can send requests inline or name a file of requests, and it can take requests from earlier `decide` records. The exact array layout is left to the builder, but it must give the head's inputs and the option logits, because Honing's diagnosis of NanoJev rested on those.

For decision models, the writer may recompute the decision head in its kernel from `read` outputs, and it may derive edits in closed form from them, e.g., the mean difference between the head inputs of correct and incorrect options. This relaxes the language-model rule against running the model's computation in the kernel. Running the backbone in the kernel stays forbidden, and so does training of any kind.

Because closed-form edits fit the practice data closely, the harness holds out part of the training split for validation:

- The harness splits the 1,200 training cases into 900 practice cases and 300 validation cases, with the same share of each workflow in both.
- `decide` and `read` passes on practice cases work as described above.
- `read` passes refuse validation cases.
- `decide` passes may run validation cases, but return only the aggregate scores and the paired comparison. They return no per-case records, so the writer cannot fit to individual validation cases.
- The writer's task prompt says that an edit must be confirmed on validation cases before the writer keeps it.
- The run report records every validation pass, so a reader can count how many times the writer checked against the validation cases.

The test split stays untouched until the final check.

The writer's task prompt gets a decision-model version. It describes these two passes and the model's architecture, in place of generation and teacher-forced reads. The language-model prompt does not change.

Each model gets its own list of protected tensors. The embeddings stay protected. The decision head's tensors are editable, because they are a likely target.

### The first task: Typed Decisions

Typed Decisions (`LocalLLaMA/typed-decisions`) has 1,200 training cases and 400 test cases. Each case is one state from one of four workflows, with five typed questions that mix Choice, Noul and Score. Each decision has a gold probability distribution and a gold label.

- The program is a DSPy module with one `Predict` per workflow. Each signature has a `state` input and one decision output field per question, with the dataset's instructions and criteria. The module routes each case to its workflow's `Predict`.
- The metric scores a case as the share of its five decisions answered correctly. A decision is correct when its most likely option equals the gold distribution's most likely option. This is the rule in the dataset's scorer (`bench/score_with_jobs.py`) for all three question types. For a Score, the metric therefore uses the level probabilities, not DSPy's decoded value. Through metric feedback, the writer also sees each decision's gold distribution and the case's Brier score, computed as the scorer computes it. Accuracy is the number we report as success.
- The writer practices on 900 training cases and validates on the other 300, as described above. The final check runs the original and edited weights on the 400 test cases.
- We also measure each model's baseline with the scorer's protocol: one request per test case, the state as parsed JSON and all five questions, sent straight to our `/v1/systemone` server without DSPy. Neither NanoJev nor d1-omni-600M is on the leaderboard (d1-2B and d1-3B are), so these baselines are new entries, comparable with the leaderboard only because they follow its protocol. DSPy's adapter wraps the state with the signature's instructions and field descriptions, so the program's baseline can differ. The write-up must report both.

### The second task: Tank War

This comes after the first success. NanoJev plays Tank War through the backend's `/v1/decide` route. A practice item is one round: a seed and a seat. The metric is the round's score: 1 for a win, 0 for a loss and 0.5 for a draw. A pass plays rounds, and the program is a DSPy module that runs `match.mjs` against the backend and returns the result. Honing's protocol run gives the comparison: a paired gain of +0.47 ± 0.12 on practice seeds, from one edit to the scalar head. The decision server needs a maximum length of 4,096 tokens, because half the rounds overflow at 2,048.

## Decisions made without the user

The user did not settle these. The builder may revisit them with a reason.

- Typed Decisions first and Tank War second.
- Serving through a local HTTP server and a subclass of `dspy.experimental.TypeSafe` that adds only save and load support, rather than an in-process client.
- Accuracy follows the dataset scorer's rule: the most likely predicted option against the most likely gold option, for every question type.
- Accuracy as the reported metric, with Brier as feedback to the writer.
- Default budget for a Typed Decisions run: 25 passes, with up to 900 items per `decide` pass and up to 1,024 requests per `read` pass. Validation passes count against the same budget.
- The validation split: 300 of the 1,200 training cases, stratified by workflow, with aggregate-only results. The user asked for held-out validation data; the size and the aggregate-only rule are my call.
- NanoJev answers Noul and Score questions as Choices.
- A System One state goes into NanoJev's `essential` section.

## Left to the builder

- Module layout, class names and file names.
- The array layout of `read` results.
- Parity tolerances, as long as they are recorded and justified.
- Batching and memory use in each framework.
- How the server finds a free port and shuts down.
- The test fixtures, including tiny random decision models for unit tests, in the style of the tiny Llama the existing tests use.

## Phases

1. Decision backends: our NanoJev and d1 loaders in PyTorch and MLX, `system_one` answering, edits, and parity tests against the reference code.
2. Serving and baselines: the `/v1/systemone` server, the Typed Decisions DSPy program and metric, and baselines for both models, with and without DSPy.
3. Writer support: the `decide` and `read` passes, the decision-model task prompt and the protected-tensor lists.
4. The first run: `WeightSpace` with `gpt-6.1-sol` on Typed Decisions for each model, with a write-up in `results/`.
5. Tank War with NanoJev.

Each phase ends with its tests passing and the language-model tests unchanged.
