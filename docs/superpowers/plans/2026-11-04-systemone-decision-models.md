# System One Decision Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `WeightSpace` on System One decision models (d1-omni-600M first, then NanoJev) with our own loaders in PyTorch and MLX, served to DSPy through its TypeSafe client, and show a test-score gain on Typed Decisions.

**Architecture:** A `DecisionBackend` owns one decision network (PyTorch or MLX) on a single worker thread, swaps edits in place, and answers System One requests. A local HTTP server exposes `/v1/systemone`. A subclass of `dspy.experimental.TypeSafe` points DSPy programs at it and records each call for the forward tool. The forward tool gains `decide` and `read` passes and a held-out validation set. The language-model path keeps its behavior.

**Tech Stack:** Python 3.12, PyTorch, MLX (`mlx`, `mlx-lm` for Qwen3 in NanoJev), DSPy 3.4.1 with `dspy[typesafe]`, `typesafe_sdk`, `safetensors`, `tokenizers`/`transformers` (tokenizers only, no remote code), pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-systemone-decision-models-design.md`

## Global Constraints

- Existing LLM behavior must not change: `HFBackend`, `MLXBackend`, `VLLMBackend`, `ForwardTool.generate`, `ForwardTool.teacher_forced`, and the LLM task prompt text. All current tests pass unchanged.
- No `trust_remote_code` and no import of Honing, Liquid or NanoJev code at run time. Reference code may run only in opt-in parity tests (`WSP_REFERENCE_TESTS=1`).
- Pinned models: `LiquidAI/d1-omni-600M` at `02b55d7076f15129e59ab3f94783f32c4b088674`; `C-Tianyu/NanoJev` at `047b927b30882a1138fc504821b82ac145a4b81a`.
- Our module attribute paths equal the checkpoint tensor names (e.g. `encoder.layers.0.conv.in_proj.weight`, `head.head.layers.0.self_attn.in_proj_weight`, `backbone.layers.0.mlp.down_proj.weight`, `scalar.weight`), so edits map by name with no translation.
- Weights load and run in float32 in both frameworks.
- Answers follow TypeSafe's response shape. Choice: `choice`, `probabilities`, `confidence = (p_max − 1/n) / (1 − 1/n)`. Score: `score` (expected level), `probabilities` keyed `"0"`..`"n-1"`, `confidence = max(0, 1 − Σ p_i |i − m| / MAD_unif)` with `MAD_unif = (1/n) Σ |i − (n−1)/2|` and `m` the most likely level. Noul: `noul` = P(true).
- Writer for decision-model examples: `dspy.LM("openai/gpt-6.1-sol")`, with no `max_tokens` and no `temperature`.
- Typed Decisions: train 1,200 cases split 900 practice / 300 validation, stratified by `workflow`, seed 0. Test 400 cases, used only in the final check.
- Accuracy rule (from the dataset's `bench/score_with_jobs.py`): a decision is correct when argmax of predicted probabilities equals argmax of the gold distribution, for every question type.
- Default decision-run budget: 25 passes; `decide` up to 900 items; `read` up to 1,024 requests.
- Only one model job runs on the Mac at a time.

## Review Focus

1. State or option text containing `<|...|>`. Expected: d1 escapes it to `<¦...¦>` so it cannot forge a delimiter. Test in Task 2.
2. A state too long for the context. Expected: d1 truncates the state on the right (its reference behavior); NanoJev returns a request error, not a silent cut. The server answers HTTP 400 and DSPy raises, so the forward tool records a program error. Tests in Tasks 2, 9 and 5.
3. An edit swap while requests are running. Expected: the swap waits until in-flight requests finish, and every answer comes from one set of weights. Test in Task 4.
4. Noul orientation. Expected: `noul` is P(true) for both models, although d1 reads options as [false, true] and NanoJev uses logits [0, z]. Tests in Tasks 2 and 9.
5. Reloading a saved program in a new process. Expected: the backend is rebuilt, the patch applied, and a new server started on a new port. Test in Task 5.

---

### Task 1: System One schema and answers

**Files:**
- Create: `src/dspy_wsp/decision/__init__.py`, `src/dspy_wsp/decision/schema.py`
- Test: `tests/test_decision_schema.py`

**Interfaces:**
- Produces:
  - `Question(type: str, instructions: str, criteria: dict | list | None)` dataclass; `parse_question(q: dict) -> Question` raising `RequestError` (re-export of `dspy_wsp.forward.RequestError`) on bad input.
  - `Question.option_keys -> list[str]`: choice keys in order; score `["0", ..., "n-1"]`; noul `["false", "true"]`.
  - `parse_request(body: dict) -> tuple[object, dict[str, Question]]` for `{"state", "questions"}` (extra keys `model`, `instructions` ignored).
  - `make_answer(q: Question, probs: list[float]) -> dict` in TypeSafe shape (Global Constraints), `probs` ordered as `option_keys`.
  - `choice_confidence(probs) -> float`, `score_confidence(probs) -> float`.

- [ ] **Step 1: Write failing tests** `test_choice_confidence_values` (`[1,0,0]`→1.0, uniform→0.0, `[0.5,0.25,0.25]`→0.25), `test_score_confidence_values` (`[0,0.5,0.5]`→0.25, `[0.5,0,0.5]`→0.0, peak→1.0), `test_make_answer_shapes` (choice keys `choice`/`probabilities`/`confidence`; score has string keys and `score == Σ i·p_i`; noul has only `noul` equal to the `"true"` probability), `test_parse_rejects_bad_questions` (choice with 1 option, score with 11 levels, unknown type).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_decision_schema.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `schema.py`. Limits: choice 2 to 255 options, score 2 to 10 levels, noul criteria optional `{"true","false"}` (also accept `"yes"`/`"no"` keys, mapped to true/false).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: System One schema and TypeSafe answer shapes`.

### Task 2: d1 encoding and PyTorch network

**Files:**
- Create: `src/dspy_wsp/decision/d1.py` (encoding, shared by both frameworks), `src/dspy_wsp/decision/d1_torch.py`
- Create: `tests/decision_fixtures.py` (tiny random d1 checkpoint folder: config like the real one with `hidden_size` 64, 4 layers `[conv, full_attention, conv, full_attention]`, 4 heads, 2 kv heads, `head_layers` 1, a tokenizer built with `tokenizers` that includes `<|reserved_7|>`..`<|reserved_11|>` and `<|mask|>` as special tokens, and temperatures)
- Test: `tests/test_d1.py`

**Interfaces:**
- Consumes: `Question` (Task 1).
- Produces:
  - `d1.encode(tok, state, q: Question, max_len: int) -> tuple[list[int], list[int]]` (token ids, marker positions), ported from the reference `prompt.encode` for the text path: `<bos> <state> state <q> instructions (<opt> <mask> text </opt>)* <decide>`, escape `<|x|>`→`<¦x¦>`, option texts from `render_options`, noul read as `[false, true]`, per-option budget `max(96, min(24k + 32, max_len // 2))`.
  - `d1.temperature_key(q) -> str` and `d1.load_config(path) -> dict`.
  - `D1Torch(nn.Module)` with submodules `encoder` (`Trunk`) and `head` (`DecisionHead`) named as the checkpoint; `D1Torch.from_dir(path, device="cpu") -> D1Torch` loading only `encoder.*` and `head.*` tensors; `logits(rows: list[tuple[list[int], list[int], Question]]) -> list[np.ndarray]` (raw option logits in the model's order, before temperature); `capture(rows, layers: list[int]) -> dict` returning per row the trunk output at the markers, the scorer input at the markers, and the hidden state after each requested trunk layer averaged over real tokens.
  - `d1.probabilities(logits: np.ndarray, q: Question, temps: dict) -> list[float]` applying the calibrated temperature and softmax, returned in `q.option_keys` order (noul reordered to `[false, true]`).
- Head layer details: `nn.TransformerEncoderLayer(d, d // 64, 4 * d, 0.0, batch_first=True, norm_first=True)` semantics (ReLU, LayerNorm eps 1e-5), key padding mask on padded text positions; scorer `LayerNorm → Linear → GELU (erf) → Linear`; type embedding added to every position before the head. Trunk: text-only path of the reference (no media prefix), centred 3-tap depthwise convolution, bidirectional attention with q/k RMSNorm and RoPE (theta from config), padding masked.

- [ ] **Step 1: Write failing tests**: `test_encode_layout` (first id is BOS, markers point at `<|mask|>` ids, one marker per option, `<|decide|>`-token last), `test_encode_escapes_delimiters` (state `"<|mask|>"` produces no extra mask id), `test_encode_truncates_long_state` (10,000-word state with `max_len=128` gives `len(ids) <= 128` and all markers inside), `test_torch_logits_shapes` (tiny fixture: one logit per option, finite), `test_noul_probabilities_order` (with logits `[0, 5]` for `[false, true]`, `probabilities(...)` gives P(true) > 0.99 in position 1 of `option_keys`), `test_padding_invariance` (a row's logits are equal alone and batched with a longer row, `atol=1e-5`).
- [ ] **Step 2:** Run `pytest tests/test_d1.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `d1.py` and `d1_torch.py`. Batch rows by padding to the longest; run in `torch.no_grad()`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: d1 encoding and PyTorch network`.

### Task 3: d1 MLX network and parity

**Files:**
- Create: `src/dspy_wsp/decision/d1_mlx.py`
- Create: `tests/test_d1_reference.py` (opt-in)
- Test: `tests/test_d1.py` (add MLX tests, skipped without `mlx`)

**Interfaces:**
- Consumes: `d1.encode`, `d1.probabilities`, the fixture.
- Produces: `D1MLX` with the same methods as `D1Torch` (`from_dir`, `logits`, `capture`), parameters stored under the same names (`mlx.utils.tree_flatten(model.parameters())` keys equal the checkpoint names), and `load_named(weights: dict[str, mx.array])` to replace parameters by name.

- [ ] **Step 1: Write failing tests**: `test_mlx_matches_torch_logits` (tiny fixture, 3 requests mixing all types, `atol=1e-4`), `test_mlx_param_names_match_checkpoint`. In `tests/test_d1_reference.py`, `test_reference_parity` (skipped unless `WSP_REFERENCE_TESTS=1`): download the pinned d1 revision, load the reference `D1OmniModel` with `trust_remote_code=True` on CPU, take 20 test cases of Typed Decisions, and require our torch and MLX probabilities to match the reference's `probabilities_batch` within `atol=1e-4`.
- [ ] **Step 2:** Run `pytest tests/test_d1.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement `d1_mlx.py` with `mlx.nn` (write attention with `mx.fast.scaled_dot_product_attention`, depthwise conv as three shifted multiplies as in the reference).
- [ ] **Step 4:** Run `pytest tests/test_d1.py -q` (PASS) and `WSP_REFERENCE_TESTS=1 pytest tests/test_d1_reference.py -q` (PASS; record the largest difference in the commit message).
- [ ] **Step 5:** Commit `feat: d1 MLX network; parity with the reference`.

### Task 4: DecisionBackend and D1Backend

**Files:**
- Create: `src/dspy_wsp/backends/decision.py`
- Modify: `src/dspy_wsp/__init__.py` (export `D1Backend`, later `NanoJevBackend`)
- Test: `tests/test_decision_backend.py`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces:
  - `DecisionBackend(WeightBackend)` with class attribute `kind = "decision"` and `concurrency = 16`. One worker thread owns the network (MLX streams belong to their thread, as in `MLXBackend`). Requests queue to it, and it answers queued requests together with one `logits` call. Methods:
    - `system_one(state, questions: dict) -> dict` returning `{"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}`, and `system_one_batch(requests: list[tuple]) -> list[dict]`.
    - `read(requests: list[dict], layers: list[int]) -> tuple[dict[str, np.ndarray], dict]`: arrays per question row, padded to the most options: `option_logits` (rows, k), `probabilities` (rows, k), `head_inputs` (rows, k, d), optional `hidden_mean` (rows, len(layers), d), `option_mask` (rows, k); `meta` lists each row's request index, question name, type and option keys, plus a `conventions` dict.
    - `_apply(edits)`: restores previously changed parameters from saved originals, then writes the edits (float32, reshaped to the checkpoint shape). Runs on the worker thread after in-flight requests finish.
    - `editable_names()`: checkpoint names that map to a loaded parameter (d1 excludes `vision.*` and `audio.*`).
    - `default_protected() -> tuple[str, ...]`, `tokenizer_dir -> Path`, `describe()`, `dims()`, `num_tokens(text)`, `render_chat` (raises: decision models have no chat), `lm()` (Task 5), `teacher_forced` and `pooled_activations` (raise `NotImplementedError` with a pointer to `read`).
  - `D1Backend(model="LiquidAI/d1-omni-600M", *, revision="02b55d7076f15129e59ab3f94783f32c4b088674", framework="mlx" | "torch", device=None, max_length=None)`. It downloads only `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and `LICENSE`. `default_protected()` is `(r"^encoder\.embed_tokens\.weight$", r"^encoder\.embedding_norm\.weight$")`. `read` returns `head_inputs` as the scorer input at each marker, and adds `trunk_markers` (rows, k, d).
  - `_init_kwargs` and `patch_dir` as in the other backends, for save and load.

- [ ] **Step 1: Write failing tests** with the tiny fixture: `test_system_one_answers_all_types`, `test_edit_changes_and_restores` (edit `head.scorer.3.weight`, answers change; `set_weights(None)` gives answers identical to before), `test_swap_waits_for_running_requests` (8 threads answering while `set_weights` runs; every result equals either the all-original or the all-edited answer for its request), `test_read_shapes_and_logits_match_answers` (softmax of `option_logits` with temperatures equals the answers' probabilities), `test_editable_names_exclude_media_and_include_head`. Parametrize over `framework` in `("torch", "mlx")`, skipping MLX when unavailable.
- [ ] **Step 2:** Run `pytest tests/test_decision_backend.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and the whole suite. Expected: PASS, with existing tests unchanged.
- [ ] **Step 5:** Commit `feat: DecisionBackend and D1Backend`.

### Task 5: Local System One server and DSPy client

**Files:**
- Create: `src/dspy_wsp/decision/server.py`, `src/dspy_wsp/decision/client.py`
- Modify: `pyproject.toml` (extra `decision = ["dspy[typesafe]", "typesafe-sdk==<pinned>", "mlx-lm>=0.32; sys_platform == 'darwin'"]`; add `typesafe-sdk` to `dev`)
- Test: `tests/test_decision_dspy.py`

**Interfaces:**
- Consumes: `DecisionBackend.system_one_batch`, `schema.parse_request`.
- Produces:
  - `SystemOneServer(backend, host="127.0.0.1", port=0)` with `.url`, `.start()`, `.close()`. `POST /v1/systemone` takes TypeSafe's body and returns `{"model", "answers", "usage"}`; malformed requests and requests that do not fit return 400 with `{"error"}`; `GET /v1/models` lists the backend's model id. It ignores the `Authorization` header.
  - `DecisionTypeSafe(dspy.experimental.TypeSafe)`, created by `DecisionBackend.lm()`, with `cache=False` and a placeholder API key. After each call it records `LMCall(prompt=json.dumps({"state", "questions"}), completion=json.dumps(answers), finish_reason="stop", input_tokens=usage, output_tokens=0)` with `backend._record`. `dump_state()` adds `"backend_class"`, `"backend": backend._init_kwargs` and `"patch": backend.patch_dir`. `load_state(state)` rebuilds the backend (shared through `LOADED_BACKENDS`), applies the patch, starts a new server, and returns a client pointed at it.
  - `DecisionBackend.lm()` starts the server on first use and returns the client.

- [ ] **Step 1:** Pin the SDK: install `dspy[typesafe]` into `.venv`, record the resolved `typesafe-sdk` version in `pyproject.toml`, and read the SDK's request path, headers and response parsing.
- [ ] **Step 2: Write failing tests** with the tiny fixture: `test_predict_with_choice_noul_score` (a `dspy.Predict` signature with one output of each type, `lm=backend.lm()`, returns typed values whose probabilities match `backend.system_one`), `test_calls_are_recorded` (`backend.record_calls()` captures one `LMCall` whose `prompt` parses to the request), `test_bad_request_is_400` (raw `requests.post` with a one-option choice), `test_save_and_load_rebuilds_backend` (compile-free: set an edit, set `patch_dir` from `Patch.from_edits(...).save`, `program.save`, clear `LOADED_BACKENDS`, `program.load(..., allow_unsafe_lm_state=True)`, same answers as edited).
- [ ] **Step 3:** Run `pytest tests/test_decision_dspy.py -q`. Expected: FAIL.
- [ ] **Step 4:** Implement with `http.server.ThreadingHTTPServer` on a daemon thread.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit `feat: local System One server and DSPy TypeSafe client`.

### Task 6: Forward tool passes for decision models

**Files:**
- Modify: `src/dspy_wsp/forward.py`
- Test: `tests/test_forward_decision.py`

**Interfaces:**
- Consumes: `DecisionBackend.read`, the existing `generate` machinery.
- Produces:
  - `ForwardTool(..., validation: Mapping | None = None)`. Validation ids are never written to practice files.
  - `Limits.max_reads: int = 1024` (new field; existing fields unchanged).
  - `handle` accepts `"type": "decide"` and `"type": "read"` only when `getattr(backend, "kind", "llm") == "decision"`. For such backends `"generate"` and `"teacher_forced"` are refused with a message naming `decide` and `read`. For LLM backends, `decide` and `read` are refused. LLM behavior is unchanged.
  - `decide`: the `generate` request without `max_tokens` and `activations`; summary `"type": "decide"`. Its `"problems"` may also name validation ids (`"validation": true` runs every validation id). A request may not mix practice and validation ids. A validation run writes no rows and returns only the aggregate summary and, with `"both"`, the paired result, plus `"validation": true`. Each validation run appends to `self.validation_log`.
  - `read`: `{"type": "read", "model": "original" | "edited", "requests": [{"state", "questions"}] | "from": {"pass": k, "ids": [...]}, "layers": [...]}`. `from` takes the System One requests recorded in `gen_<k>.jsonl` or `gen_<k>_<model>.jsonl` for the given practice ids. It refuses more than `max_reads` question rows and refuses `from` on validation passes (they have no rows). It writes each array to `fwd_<k>/<name>.npy` plus `meta.json`, and returns the file list with shapes, as `teacher_forced` does.
- [ ] **Step 1: Write failing tests** with a fake decision backend (subclass `DecisionBackend` with a tiny d1 fixture): `test_decide_matches_generate_rows`, `test_validation_decide_returns_aggregates_only` (no `gen_*.jsonl` written; summary has `paired` with `"both"`), `test_validation_and_practice_cannot_mix`, `test_read_from_recorded_pass`, `test_read_limit_enforced`, `test_llm_backend_refuses_decide_and_read`, `test_decision_backend_refuses_generate`.
- [ ] **Step 2:** Run `pytest tests/test_forward_decision.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the new tests and `tests/test_forward.py`. Expected: PASS, with `tests/test_forward.py` unchanged.
- [ ] **Step 5:** Commit `feat: decide and read passes, validation holdout`.

### Task 7: Optimizer and decision task prompt

**Files:**
- Modify: `src/dspy_wsp/optimizer.py`, `src/dspy_wsp/prompts.py`, `src/dspy_wsp/kernel.py`, `src/dspy_wsp/_kernel_main.py`
- Test: `tests/test_decision_optimizer.py`, plus `tests/test_prompt_snapshot.py`

**Interfaces:**
- Consumes: Tasks 4 to 6.
- Produces:
  - `WeightSpace(..., protected=None, holdout: int | float = 0, holdout_key: Callable[[dspy.Example], str] | None = None)`. With `protected=None`, the optimizer uses `backend.default_protected()` if the backend has it, else `DEFAULT_PROTECTED` (so LLM runs are unchanged). `holdout` moves that many (or that share of) trainset examples to validation, stratified by `holdout_key`, seeded by `seed`.
  - Decision backends get `prompts.build_decision_task_prompt(...)`: the same sections as the LLM prompt, with the forward-pass section describing `decide`, `read` and validation runs, the model's architecture from `backend.describe()`, and rule (b) replaced by: "running the model's backbone in the kernel; you may recompute the decision head from `read` outputs and derive edits from them in closed form". It adds: "An edit is kept only if a validation run confirms it. Validation runs return aggregate scores only." It also states the validation set's size.
  - `Kernel(..., tokenizer_dir=None)` passes `WSP_TOKENIZER_DIR` to the kernel, which loads `tok` from there when set; the optimizer passes `backend.tokenizer_dir` when present.
  - `report.json` gains `"validation_runs": forward.validation_log` for decision runs.
- [ ] **Step 1: Write failing tests**: `test_llm_task_prompt_unchanged` (a snapshot of the LLM task prompt for the `FakeBackend` in `tests/conftest.py`, taken before this task's changes and stored in `tests/data/llm_task_prompt.txt`), `test_decision_prompt_mentions_decide_read_validation`, `test_holdout_split_is_stratified_and_hidden` (practice file has 900 of 1,200 synthetic examples, 75 per workflow per 4 workflows in validation), `test_backend_default_protected_used` (NanoJev-style `norm.weight` editable when the backend's list omits it), `test_compile_runs_with_scripted_writer` (tiny d1 backend, scripted writer that saves an edit, runs a validation `decide`, and finishes; report has `validation_runs` and a final check).
- [ ] **Step 2:** Generate the snapshot from the current code, then run the tests. Expected: the snapshot test PASSES and the new ones FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the whole suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat: WeightSpace runs decision backends with a validation holdout`.

### Task 8: Typed Decisions example and baselines

**Files:**
- Create: `examples/typed_decisions.py`
- Test: `tests/test_typed_decisions_example.py`

**Interfaces:**
- Consumes: `D1Backend` (later `NanoJevBackend`), `WeightSpace`.
- Produces:
  - `load_split(split: str) -> list[dspy.Example]` reading `all/{split}-00000-of-00001.parquet` from `LocalLLaMA/typed-decisions` (cached under `examples/runs/typed-decisions/`). Each example has inputs `workflow`, `state` and labels `gold`, `questions`.
  - `TypedDecisions(dspy.Module)`: one `dspy.Predict` per workflow. Each signature has input `state: str` and one output per question, typed `Choice[(key, description), ...]`, `Noul[(True, desc), (False, desc)]` or `Score[level, ...]` with the dataset's instructions as the field description. It routes by `workflow`.
  - `metric(example, pred, trace=None) -> dspy.Prediction(score=accuracy over the five decisions, feedback=per-decision gold argmax, predicted argmax and the case's Brier score)`, using predicted probabilities (`Noul.probability` mapped to `{"true": p, "false": 1-p}`; `Score.probabilities` keyed by level).
  - `protocol_baseline(backend, cases) -> dict` sending each test case's own `{"state": parsed JSON, "questions"}` to `backend.lm()`'s server with `requests`, scored with the dataset scorer's formulas (accuracy, KL from gold, Brier, ECE with its bins).
  - CLI: `--model d1|nanojev`, `--framework mlx|torch`, `--baseline protocol|program`, `--smoke` (3 passes, 64-item passes, 50 validation cases, 40 test cases), `--passes`, `--run-dir`.
- [ ] **Step 1: Write failing tests** on three synthetic rows in the dataset's format: `test_signatures_built_per_workflow`, `test_metric_accuracy_and_feedback` (a hand-built prediction with known probabilities scores 3/5), `test_protocol_scoring_formulas` (KL, Brier and ECE on a fixed toy input match values computed by hand).
- [ ] **Step 2:** Run `pytest tests/test_typed_decisions_example.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: Typed Decisions example, metric and baselines`.

### Task 9: First run with d1-omni-600M

**Files:**
- Create: `results/2026-11-04-typed-decisions-d1/README.md` and its artifacts (as in `results/2026-11-04-math500-qwen-vllm-1h/`)

- [ ] **Step 1:** Run `WSP_REFERENCE_TESTS=1 pytest tests/test_d1_reference.py -q` if not already run on real weights. Expected: PASS.
- [ ] **Step 2:** Baselines on test with MLX: `python examples/typed_decisions.py --model d1 --framework mlx --baseline protocol`, then `--baseline program`. Record accuracy, KL, Brier and ECE for both.
- [ ] **Step 3:** Smoke run: `python examples/typed_decisions.py --model d1 --framework mlx --smoke --run-dir runs/td-d1-smoke`. Expected: exits 0, with a report holding a final check and `validation_runs`.
- [ ] **Step 4:** Full run: `python examples/typed_decisions.py --model d1 --framework mlx --run-dir runs/td-d1` (25 passes, 900/300 split, final check on 400 test cases).
- [ ] **Step 5:** Write the results README: setup, baselines, the writer's passes, validation runs, final paired result, and whether the gain exceeds twice its standard error. Commit.

### Task 10: NanoJev encoding, networks and backend

**Files:**
- Create: `src/dspy_wsp/decision/nanojev.py` (encoding), `src/dspy_wsp/decision/nanojev_torch.py`, `src/dspy_wsp/decision/nanojev_mlx.py`, `tests/test_nanojev.py`, `tests/test_nanojev_reference.py` (opt-in)
- Modify: `src/dspy_wsp/backends/decision.py` (add `NanoJevBackend`), `tests/decision_fixtures.py` (tiny NanoJev: a Qwen3 backbone with 2 layers and width 64, the head, a tokenizer)

**Interfaces:**
- Produces:
  - `nanojev.encode(tok, state, q: Question, max_len: int) -> list[list[int]]` (one path per candidate), matching `prepare_examples`: segments `f"State:\n{state}\n"` (with Python `str()` of a non-string state) and `f"Question type: {t}\nQuestion:\n{instructions}\n"` with `t` in `choice`, `score`, `boolean`, plus boolean criteria lines; each path ends `Candidate:\n{text}\nDecision:` and EOS; choice texts `f"{key}: {desc}"`, score texts are the level descriptions, boolean has one text `"The proposition is true."`. A path longer than `max_len` raises `RequestError`.
  - `NanoJevTorch` and `NanoJevMLX` with `from_dir`, `logits(rows)` and `capture(rows, layers)`; names `backbone.*`, `norm.*`, `scalar.*`, `set_project.*`, `set_attention.*` and `set_output.*`. The backbone reads the last real token of each path (right padding; causal attention). The head is `norm → scalar`, plus for choices the set correction `set_output(tanh(u + MHA(u)))` with `u = set_project([h, log k])`. A boolean's logits are `[0, z]`. The MLX backbone uses `mlx_lm.models.qwen3` with `backbone.config.json` and our names mapped onto its parameters.
  - `NanoJevBackend(model="C-Tianyu/NanoJev", *, revision="047b927b30882a1138fc504821b82ac145a4b81a", framework, device=None, max_length=8192)` downloading `best.safetensors`, `config.json`, `backbone_config/*` and `tokenizer/*` only. `default_protected()` is `(r"^backbone\.embed_tokens\.weight$", r"^backbone\.norm\.weight$")`. `read`'s `head_inputs` are the normed path ends (`h` after `norm`), with `set_delta` (rows, k) added for choices.
- [ ] **Step 1: Write failing tests**: `test_encode_matches_reference_layout` (text of the decoded path for one choice, one score and one boolean), `test_long_path_raises`, `test_boolean_logits_zero_and_z`, `test_mlx_matches_torch_logits`, `test_backend_edit_scalar_changes_answers`; in `tests/test_nanojev_reference.py` (opt-in), parity with `DecisionModel` from the pinned NanoJev source on CPU in float32, on 20 Typed Decisions test cases, `atol=1e-4`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the unit tests and the opt-in parity test. Expected: PASS.
- [ ] **Step 5:** Commit `feat: NanoJev loaders and backend`.

### Task 11: NanoJev run

- [ ] **Step 1:** Baselines as in Task 9, with `--model nanojev`.
- [ ] **Step 2:** Smoke run, then the full run, with `--model nanojev`.
- [ ] **Step 3:** Write `results/2026-11-04-typed-decisions-nanojev/README.md` and commit.

Tank War (spec phase 5) gets its own plan after the Typed Decisions results. It needs the `/v1/decide` route, `match.mjs` and a play pass.
