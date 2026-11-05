I’ll inspect the practice data and create a fixed fitting/test split before changing any weights. This keeps a portion of the practice examples independent of edit selection and establishes a reproducible baseline.

```python
# Hypothesis: systematic errors may originate in the shared decision readout; inspect schemas and create a reproducible diagnostic split.
import json, pathlib, numpy as np, torch, collections
work = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/nanojev/work")
examples = [json.loads(s) for s in (work/"practice/trainset.jsonl").read_text().splitlines()]
byid = {e["id"]: e for e in examples}
rng = np.random.default_rng(814)
ids = [e["id"] for e in examples]
rng.shuffle(ids)
fit_ids, test_ids = ids[:600], ids[600:]
print("Examples:",len(examples))
print("Workflows:",collections.Counter(e["inputs"]["workflow"] for e in examples))
print("First example:",json.dumps(examples[0],indent=2)[:14000])
(work/"notes.md").write_text(
    "# Weight-repair experiment\n"
    "No edits yet. Hypothesis to investigate: shared readout misses task-relevant directions.\n"
    "Deterministic split: 600 fitting cases, 300 untouched practice test cases (seed 814).\n"
    "Next: original baseline on all 900 practice cases.\n"
)
```
