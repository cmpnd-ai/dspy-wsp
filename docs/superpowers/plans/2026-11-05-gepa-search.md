# GEPA search for WeightSpace: implementation plan

**Goal:** `WeightSpace(..., search=GEPASearch(...))` runs GEPA's population search, with WeightSpace's Python kernel making each child edit.

**Spec:** `docs/superpowers/specs/2026-11-05-gepa-search-design.md`

**Tech stack:** gepa 0.1.4 (pinned by dspy 3.4.1), the existing kernel, session, forward-pass tool and decision backends.

## Global constraints

- `search=None` leaves today's behavior and today's task prompt unchanged. `tests/test_prompt_snapshot.py` must still pass.
- Files in `src/dspy_wsp/edit_gepa/` do not change.
- GEPA search needs a decision backend and `holdout > 0`.
- Every forward pass counts against one master budget.

## Review focus

1. A round whose writer saves no edit, or saves an invalid one. Expected: the child equals the parent, or the round is dropped. The search does not crash.
2. Restoring the parent. The edit folder must hold exactly the parent's tensors when a round starts, including when the parent has fewer tensors than the last child.
3. Budget overrun. The stop rule must leave room for the worst-case round.
4. The writer finishing a round without using its passes. The unused passes stay in the master budget.
5. Validation passes inside a round are refused and not counted.

---

### Task 1: Settings, checks and prompt variant

**Files:** `src/dspy_wsp/search.py` (new), `src/dspy_wsp/optimizer.py`, `src/dspy_wsp/prompts.py`, `src/dspy_wsp/__init__.py`, `tests/test_gepa_search.py` (new)

**Produces:** `GEPASearch(min_rounds=10, minibatch_size=8, passes_per_proposal=2, code_runs_per_proposal=40)` with `round_cost` (`3 + passes_per_proposal`) and `min_passes()` (`1 + min_rounds * round_cost`). `WeightSpace(search=...)` raises `ValueError` for a non-decision backend, for `holdout == 0`, and for `max_passes < search.min_passes()`. `build_decision_task_prompt(..., search_round=False)`: when true, the validation pass entry, the validation advice and the ending describe a GEPA round.

Tests:
- `test_gepa_settings_and_checks`
- `test_round_prompt_drops_validation_passes`
- the snapshot test is unchanged.

### Task 2: Validation refusal in rounds

**Files:** `src/dspy_wsp/forward.py`

**Produces:** `ForwardTool.allow_validation` (default True). When false, a validation decide raises `RequestError` and is not counted.

Test: `test_validation_refused_when_disallowed`

### Task 3: The search

**Files:** `src/dspy_wsp/search.py`

**Produces:** `run_gepa(...) -> (edits, final_answer, search_report)`, called by `WeightSpace.compile`. Its parts:

- **The adapter.** `evaluate(batch, candidate, capture_traces)` loads the candidate's edits, runs `ForwardTool.run_items` under `backend.using(edits)` and counts one pass. Its feedback is one line per case, with the id, the score and the metric's feedback.
- **The proposer.** It restores the parent's edits and notes into the work folder and runs a `Session` with a round budget, `max_nudges=0` and the round's opening message. It then adds the round's usage to the master budget and saves the result as candidate `cNNN`.
- **The stop callback.** It stops when the passes left are fewer than `round_cost`, or when the code runs are used up.
- **The logging callback.** It writes `rounds.jsonl`, rewrites `progress.md` and logs one line per round.

Tests, with the scripted writer and the tiny d1-omni fixture:
- `test_gepa_round_flow`
- `test_gepa_budget_and_stop`
- `test_gepa_logs`
- `test_parent_restored_exactly`
- `test_round_without_edit`

### Task 4: Wiring, report and example

**Files:** `src/dspy_wsp/optimizer.py`, `examples/typed_decisions.py`, `examples/procedural_decisions.py`, `README.md`

**Produces:**
- `compile` branches on `self.search`. The report gains `search`.
- The examples take `--search {conversation,gepa}` and `--rounds`. For GEPA, the default passes are `GEPASearch.min_passes()`. The run folder is named `<model>-gepa`.
- The README gets a short section on the new search.

Test: the full suite passes.
