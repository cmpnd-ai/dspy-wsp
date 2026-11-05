I’ll inspect the practice-data structure and workflow balance, check for existing edits, and start an experiment log. This establishes an untouched baseline before choosing a failure mechanism to investigate.

```python
# Hypothesis: errors cluster by question/option; inspect the schema before measuring and localizing them.
import os, json, pathlib, collections, numpy as np, torch
work = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1/work")
examples = [json.loads(x) for x in (work/"practice/trainset.jsonl").read_text().splitlines()]
byid = {x["id"]: x for x in examples}
print(len(examples), collections.Counter(x["inputs"]["workflow"] for x in examples))
print(json.dumps(examples[0], indent=2)[:6500])
notes = work/"notes.md"
notes.write_text("# Weight-repair experiment\nNo edits yet. Plan: baseline all 900 practice cases; inspect errors and read decision-head representations. Reserve a deterministic half of practice for testing closed-form edits.\n")
print("Existing edits:", edited())
print("First IDs:", [x["id"] for x in examples[:5]])
```
