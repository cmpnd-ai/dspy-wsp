I’ll inspect the practice data and available head tensors, then record the initial plan. This establishes the task structure before measuring baseline failures; no weights are changed.

```python
# Hypothesis: baseline errors will reveal a rule-following failure that can be repaired in the decision head.
import os, json, collections, numpy as np, torch
from pathlib import Path
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/procedural-decisions/d1/work")
data = [json.loads(s) for s in (work/"practice/trainset.jsonl").read_text().splitlines()]
print(len(data), collections.Counter(x["inputs"]["workflow"] for x in data))
print(json.dumps(data[0], indent=2)[:6500])
(work/"notes.md").write_text("Initial diagnosis. No edits. Next: original baseline on all 900 practice examples.\n")
print("tensor head names", [n for n in NAMES if n.startswith("head.")])
```
