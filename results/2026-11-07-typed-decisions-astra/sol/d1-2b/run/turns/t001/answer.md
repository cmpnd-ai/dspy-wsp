I will inspect the practice schema and create a reproducible fit/holdout split before measuring the original model. This keeps edit selection separate from practice confirmation.

```python
# Hypothesis: failures can be localized by workflow and question before choosing an intervention.
import os, json, numpy as np, torch, collections, random
from pathlib import Path
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-2b/work")
data = [json.loads(s) for s in (work/"practice/trainset.jsonl").read_text().splitlines()]
rng = random.Random(718)
bywf = collections.defaultdict(list)
for ex in data: bywf[ex["inputs"]["workflow"]].append(ex)
fit, hold = [], []
for wf, xs in sorted(bywf.items()):
    rng.shuffle(xs)
    fit += xs[:len(xs)//2]
    hold += xs[len(xs)//2:]
fitids = [x["id"] for x in fit]
holdids = [x["id"] for x in hold]
def note(s):
    with (work/"notes.md").open("a") as f: f.write(s+"\n")
note("# Weight repair session\nNo edits. Stratified random split seed 718; fit=%d, holdout=%d.\nNext pass: original DECIDE on all 900 practice examples."%(len(fit),len(hold)))
json.dump({"type":"decide","model":"original","problems":[x["id"] for x in data]},open(work/"baseline_request.json","w"))
print("workflow counts:", {k:len(v) for k,v in bywf.items()})
print("example:", json.dumps(data[0],indent=2)[:11000])
print("existing edits:", edited())
```