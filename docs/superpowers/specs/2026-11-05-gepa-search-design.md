# GEPA search for WeightSpace: design

## Intent

WeightSpace runs one long conversation. The writer reads weights in a Python kernel, spends forward passes on practice cases, and saves one edit at the end. `EditGEPA` (in `dspy_wsp.edit_gepa`) runs a population search instead. GEPA keeps a set of candidate networks, picks a parent from its Pareto frontier, asks the writer for one change, and keeps the child only if it beats the parent on a small batch of practice cases.

This design adds GEPA's population search to WeightSpace and keeps WeightSpace's Python kernel as the way the writer makes each change. Both searches then share the model, the practice and validation cases, the writer, the protected tensors, the forward-pass budget and the final test. A run with each search on Typed Decisions or Procedural Decisions compares the two search methods and nothing else.

## Goals

- `WeightSpace(..., search=GEPASearch(...))` runs a GEPA population search over saved edits. Leaving `search` unset keeps today's single conversation.
- The forward-pass budget counts every pass, whether the writer runs it or GEPA does.
- A GEPA run can be configured to guarantee a minimum number of rounds (10 for the first comparison).
- The run writes a log line for every round as it happens, so a reader can see when improvements land.
- The benchmark scripts take `--search gepa`.

## Non-goals

- Changes to the files in `dspy_wsp.edit_gepa`. Its text-edits search stays as its author wrote it.
- GEPA search for LLM backends. GEPA selects on a validation set, and only decision backends have one (`holdout`).
- GEPA's merge step.

## How a GEPA run works

1. **Setup.** Everything WeightSpace does before the session stays the same: the practice and validation split, the edit store, the forward-pass tool, the task prompt and the kernel.
2. **Seed.** Candidate `c000` is the original model, with no edits. GEPA scores it on the validation set.
3. **A round.**
   - GEPA picks a parent from its Pareto frontier and scores it on a minibatch of practice cases, with the metric's feedback.
   - The writer gets a short session that starts from the parent: the parent's edits are in the edit folder and the parent's notes are in `notes.md`. Its first message holds the round number, the parent's edit summary and the minibatch feedback.
   - The session has its own small budget of forward passes and Python runs. The writer may finish early. It cannot run validation passes, because GEPA uses the validation set to select candidates.
   - When the session ends, the edit folder and `notes.md` are saved as a new candidate.
   - GEPA scores the child on the same minibatch. If the child scores higher than the parent, GEPA scores it on the validation set and adds it to the population.
4. **Stop.** The search stops when the passes left can't pay for one more round.
5. **Finish.** The candidate with the best validation score becomes the result. The rest is WeightSpace's existing tail: the final paired check on `valset`, the patch and the report.

The kernel and the work folder are shared across rounds. Read outputs, helper files and kernel variables from earlier rounds stay available. Each round only swaps the edit folder and `notes.md`.

## Budget

- One master `Budget` holds `max_passes` and `max_code_runs`.
- Each GEPA scoring counts one pass. That matches WeightSpace, where a validation run over the whole validation set counts one pass.
- A round costs at most `3 + passes_per_proposal` passes: the parent's minibatch, the child's minibatch, the child's validation run and the writer's passes.
- The seed's validation run costs one pass.
- `WeightSpace` refuses a configuration whose `max_passes` is below `1 + min_rounds * (3 + passes_per_proposal)`.
- With `passes_per_proposal=2` and `min_rounds=10`, that minimum is 51 passes. A rejected child skips its validation run, so a run usually gets more than 10 rounds.
- The comparison's single-conversation run gets the same `max_passes`.

## API

```python
from dspy_wsp import GEPASearch, WeightSpace

optimizer = WeightSpace(
    metric=metric, writer_lm=writer, backend=backend, max_passes=51, holdout=300,
    search=GEPASearch(min_rounds=10, minibatch_size=8, passes_per_proposal=2, code_runs_per_proposal=40),
)
compiled = optimizer.compile(program, trainset=train, valset=test)
```

`GEPASearch` is a dataclass of settings. `compile` returns the same program, patch and report as today. The report gains a `search` section with the rounds, the candidates and the best candidate.

## Logging

Each run directory gets:

- `rounds.jsonl`: one row per round, written when the round ends. A row holds the parent, the parent's and the child's minibatch scores, whether the child was accepted, the child's validation score, the best validation score so far, passes and Python runs used, the writer's tokens and cost so far, and the elapsed time.
- `progress.md`: a table of the rounds, rewritten after each round.
- `candidates/cNNN/`: each candidate's `edited/` folder, `notes.md` and the writer's final message.
- `rounds/rNN/`: each round's session turns and transcript.
- One `logging` line per round, so the console log of a long run shows progress.

## Tests

Tests use the tiny d1-omni fixture and the scripted writer that the existing optimizer tests use. They check:

- the round flow: parent edits are restored, a child is saved, and acceptance follows the minibatch scores;
- pass accounting across GEPA scorings and the writer's passes;
- the minimum-rounds check;
- the stop rule;
- refusal of validation passes inside a round;
- the log files;
- that `search=None` behaves as before.

## Calls made without asking

- Each GEPA scoring counts one pass, as a validation run does today.
- The writer may finish a round early. Unused round passes return to the master budget.
- Validation passes are refused inside rounds.
- Each child inherits its parent's notes and the shared kernel.
- GEPA skips the writer session when the parent scores perfectly on its minibatch, because a child can't beat it there.
- A writer failure, or a scoring where every example fails, stops the search. The run finishes with the best candidate so far.
- The session retries a writer call twice after a transport error, such as a read timeout.
- `resume=True` continues a stopped search from GEPA's saved state and `rounds.jsonl`. GEPA scores the seed again when it starts, so the search returns the seed's saved scores for that scoring and counts no pass.
- `code_runs_per_proposal` defaults to 30, so 10 rounds fit the default 300 Python runs.
