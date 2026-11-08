I’ll inspect the practice schemas and records, then measure the unedited model’s errors. That will let me target a specific failure rather than change confidence globally.
The first inspection will check workflow balance, option names, and available head tensors. I’ll also create the experiment log before running any forward passes.
```python
# Hypothesis: workflow/question-specific errors, rather than global calibration, will identify a repairable head mechanism.
import os, json, math, random, pathlib, collections, statistics
import numpy as np
import torch

WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-astra-strict/work")
PRACTICE = WORK / "practice" / "trainset.jsonl"
examples = [json.loads(s) for s in PRACTICE.read_text().splitlines() if s.strip()]
by_id = {str(e["id"]): e for e in examples}
NOTE = WORK / "notes.md"
def note(text):
    with NOTE.open("a") as f:
        f.write(text.rstrip() + "\n\n")
note("# Weight repair experiments\nNo edits yet. All analyses use practice data only; validation will be used only for aggregate confirmation.")
print("Examples:", len(examples), "workflows:", collections.Counter(e["inputs"]["workflow"] for e in examples))
print("Current edits:", edited())
print("Head tensors:", {n: SHAPES[n] for n in NAMES if n.startswith("head.")})
for wf in sorted({e["inputs"]["workflow"] for e in examples}):
    ex = next(e for e in examples if e["inputs"]["workflow"] == wf)
    print("\nWORKFLOW", wf)
    print(json.dumps(ex, ensure_ascii=False)[:5600])
```
