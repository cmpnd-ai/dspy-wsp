# GEPA over weight edits: MinAtar Seaquest

`dspy_wsp.edit_gepa.EditGEPA` runs a GEPA population search in which a writer LM edits a small network's weights directly. Each network is an *edits program*: assignments to slices of named tensors, in the language of the paper's Experiment 5, applied to all-zero weights. The writer reads how a network behaved and writes one edits block. GEPA keeps the children that score higher. There are no gradients and no training.

This folder is a recipe: it applies `EditGEPA` to MinAtar Seaquest, starting from the network the paper's writer built by hand in Experiment 5. That network scored 53.1 on its own.

## How a search works

Networks are compared on a validation set split into instances. Each iteration:

1. **Pick a parent** from the Pareto frontier: networks that are best on at least one validation instance, weighted by how many instances they win.
2. **Evaluate it** on a minibatch of training instances. Your `evaluate` function also returns feedback text for the writer.
3. **Ask the writer** for one edits block. It sees the parent's full program, your `context` text, and the feedback. The child is the parent's program plus the block, or the block alone if its first line is `# base: zero`.
4. **Check the child** on the same minibatch, which with seeded episodes cancels most noise. If the child doesn't score higher, it is dropped.
5. **Admit it**: evaluate it on the validation set and add it to the population.

The search stops before the evaluation budget runs out and returns the network with the best validation mean. The budget counts evaluation units, here episodes, so runs and baselines can be compared at matched forward passes. GEPA's merge step is off.

The writer has no memory between calls. It knows only the parent's program, including the comments the writer wrote in earlier blocks, the context and the latest feedback.

## The API

```python
import dspy
from dspy_wsp.edit_gepa import EditGEPA

shapes = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}

def evaluate(weights, instances, with_feedback):
    """weights: name -> np.ndarray. Return one score per instance and, if asked, feedback text."""
    ...
    return scores, feedback_text if with_feedback else None

search = EditGEPA(
    writer_lm=dspy.LM("openai/gpt-6-astra", reasoning_effort="xhigh", max_tokens=100_000, cache=False),
    shapes=shapes,
    evaluate=evaluate,
    task=task_prompt,          # the task, the network and the edits language (system prompt)
    context=design_notes,      # optional, shown before every parent program
    budget=6400,               # evaluation units for the whole search
    units_per_instance=8,      # e.g. an instance is a group of 8 episodes
    minibatch_size=8,
    prices=(10.0, 1.0, 50.0),  # $/M tokens: input, cached input, output (optional)
    run_dir="runs/my-search",
)
result = search.run(seed_program, trainset=train_instances, valset=val_instances)
result.best_program, result.best_weights, result.best_val, result.writer_usd
```

Use `cache=False` on the writer LM, so that a retried prompt gets a fresh reply. `dspy_wsp.edit_gepa.language.apply(program, shapes)` turns a program into weights. Statements it can't read are skipped and listed, and the writer sees that list in its next feedback. Programs are parsed with `ast` and never executed.

What gets written to `run_dir`:

```
proposals.jsonl   every writer call: parent id, the feedback it read, its reply, tokens and $ so far
evals.jsonl       every evaluation: program id, split (train/val), mean score, units used so far
programs.jsonl    every program evaluated on the validation set: id, full text, validation mean
best_program.txt  the selected program
result.json       best_val, units_used, writer calls, tokens and $, seconds
gepa_state/       GEPA's own state
```

## Run the recipe

```bash
pip install -e .                                   # from the repo root
pip install -r examples/seaquest_gepa/requirements.txt
cd examples/seaquest_gepa
python fetch_data.py                               # the paper's Seaquest files, pinned and hash-checked
python check.py                                    # replication gate, ~5 s on a laptop
python run_gepa.py --dry-run                       # free: the writer is a stub, 700 episodes
python run_gepa.py --seed 0                        # the real search
```

`check.py` rebuilds the paper's network from its edits program and confirms it is identical to the released weights. It then scores the network on the 1,000 test episodes; expect 53.33 (SE 1.33) against the paper's 53.11 (SE 1.36).

The writer defaults to `openai/gpt-6-astra` at `xhigh` effort, the paper's model. For an Azure deployment, put `AZURE_API_BASE` (`https://<resource>.services.ai.azure.com/openai/v1`) and `AZURE_API_KEY` in `examples/seaquest_gepa/.env`. Any other `dspy.LM` model works with `--lm` and `--effort`. Set `WG_PROCS` to the number of processes that play episodes.

The recipe's settings:

| Setting | Value |
|---|---|
| Network | 600 → 32 tanh → 32 tanh → 6, 20,486 parameters; the paper's architecture and 6-channel input |
| Episodes | Actions sampled from the softmax, sticky actions 0.1, cut at 10,000 steps |
| Instance | A group of 8 seeded episodes |
| Validation | 16 groups (128 episodes) |
| Minibatch | 8 groups (64 episodes) |
| Budget | 6,400 episodes, the same as 50 passes of 128 |
| Test | 1,000 episodes, never seen by the search |

The feedback is behaviour only:
- score statistics and how the episodes ended;
- enemies shot, divers picked up and surfacings;
- the executed-action mix and the time spent in each row;
- the last 5 steps of the 2 lowest-scoring episodes. Each step shows the grid as letters, the action probabilities, and the sampled and executed action.

The writer never sees activations.

## Pilot result

One seed, before this code was packaged as an API. Details are in [results/2026-11-05-seaquest-gepa-pilot](../../results/2026-11-05-seaquest-gepa-pilot/).

| | Test score (1,000 episodes) |
|---|---|
| Paper's hand-written network (start) | 53.3 ± 1.3 |
| EditGEPA, 6,272 episodes, 37 writer calls | **85.6 ± 1.1** (paired gain +32.3 ± 1.8) |

The search cost $36.26 of writer tokens and took 5 h 21 min, most of it waiting on the writer. Testing took 1.4 min. The best network was found after 1,536 episodes and 7 writer calls ($6.61); the remaining 30 calls found nothing better. For comparison, a single conversation with the same writer and budget reached 135.6 in the same pilot; that arm isn't part of this API.

## Credits and licenses

The Seaquest network, the writer's Experiment 5 reply and prompt, and the evaluation file come from Takehi et al.'s [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (CC BY 4.0). `fetch_data.py` downloads them at a pinned commit instead of copying them into this repo, because the Experiment 5 prompt contains MinAtar's source code (GPL-3.0). MinAtar itself is installed from PyPI.
