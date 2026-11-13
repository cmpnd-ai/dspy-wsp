I’ll first inspect the practice records and question schemas, then establish an unedited baseline. That will show which failures are common enough to justify a targeted repair.

```python
# Hypothesis: baseline errors will cluster by question or option, revealing a specific decision-head failure.
import os, json, math, time, collections, pathlib, random
import numpy as np
import torch
WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/strands-2b-astra/work")
PRACTICE = WORK / "practice" / "trainset.jsonl"
examples = [json.loads(line) for line in open(PRACTICE)]
byid = {str(ex["id"]): ex for ex in examples}
print("Examples:", len(examples), "Workflows:", collections.Counter(ex["inputs"]["workflow"] for ex in examples))
print("Existing edits:", edited())
for wf in sorted({ex["inputs"]["workflow"] for ex in examples}):
    ex = next(ex for ex in examples if ex["inputs"]["workflow"] == wf)
    print("\nWORKFLOW", wf, "ID", ex["id"])
    print(json.dumps(ex, ensure_ascii=False, indent=2)[:6800])
notes = WORK / "notes.md"
notes.write_text("# Weight-repair experiment\n\n"
                 "Baseline: no edits. Plan: diagnose option/question failures, localize them in the pointer head, "
                 "derive only closed-form targeted edits, test strengths, then confirm on untouched practice and validation.\n")
def lognote(text):
    with open(notes, "a") as f:
        f.write("\n" + text + "\n")
baseline_req = {"type": "decide", "model": "original", "problems": [ex["id"] for ex in examples]}
json.dump(baseline_req, open(WORK / "baseline_request.json", "w"))
lognote("## Before pass 1\nExact edit: none. Original baseline on all 900 practice IDs, listed in baseline_request.json. "
        "No examples have yet been used to fit or select edits.")
print("\nBaseline request saved:", WORK / "baseline_request.json")
```
