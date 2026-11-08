I’ll first inspect the practice labels and available tensors, then run an unedited baseline to identify the largest recurring errors before choosing an intervention.
```python
# Hypothesis: the baseline's errors will cluster by question and option, exposing a specific mechanism to investigate.
import os, json, math, time, collections, pathlib
import numpy as np
import torch

WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-2b-astra/work")
PRACTICE = WORK / "practice" / "trainset.jsonl"
examples = [json.loads(s) for s in PRACTICE.read_text().splitlines() if s.strip()]
by_id = {e["id"]: e for e in examples}
NOTES = WORK / "notes.md"
def note(text):
    with NOTES.open("a") as f:
        f.write(text.rstrip() + "\n\n")
note("# Weight-edit experiment\n\nGoal: diagnose typed-decision failures and make a targeted, closed-form block-weight repair. No backbone execution outside the forward tool; no gradient fitting.\n\nInitial edit: none.")
print("Examples:", len(examples), "workflows:", dict(collections.Counter(e["inputs"]["workflow"] for e in examples)))
print("Current edits:", edited())
print("First example:", json.dumps(examples[0], indent=2)[:8500])
print("Last-block tensors:", [(n, SHAPES[n]) for n in NAMES if ".layers.23." in n])
baseline_file = WORK / "baseline_request.json"
baseline_file.write_text(json.dumps({"type":"decide", "model":"original", "problems":[e["id"] for e in examples]}))
note("Planned forward pass 1: DECIDE original, all 900 practice IDs from trainset.jsonl. Current edit: none. Purpose: baseline diagnosis; no examples used for fitting yet.")
print("Baseline request:", str(baseline_file))
```
