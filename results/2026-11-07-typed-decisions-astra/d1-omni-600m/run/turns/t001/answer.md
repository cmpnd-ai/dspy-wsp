I’ll inspect the practice data and model configuration, then establish an unedited baseline. I’ll reserve a practice subset for a paired test so the final edit is judged on examples not used to design it.
```python
# Hypothesis: workflow-specific error patterns can guide a targeted head repair rather than a global calibration change.
import os, json, math, time, random, collections, pathlib
import numpy as np
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(min(8, os.cpu_count() or 1))
WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-astra/work")
MODEL = pathlib.Path("/Users/dbreunig/.cache/huggingface/hub/models--LiquidAI--d1-omni-600M/snapshots/02b55d7076f15129e59ab3f94783f32c4b088674")
practice = [json.loads(s) for s in (WORK/"practice/trainset.jsonl").read_text().splitlines() if s.strip()]
by_id = {str(x["id"]): x for x in practice}
def note(text):
    with open(WORK/"notes.md", "a") as f:
        f.write(text.rstrip()+"\n\n")
print("Examples:", len(practice), "Workflows:", collections.Counter(x["inputs"]["workflow"] for x in practice))
print("Example:", json.dumps(practice[0], indent=2)[:7000])
print("Model files:", [p.name for p in MODEL.iterdir()])
print("Existing edits:", edited())
print("Tensor count:",len(NAMES))
rng = random.Random(82149)
split_ids = {}
for wf in sorted({x["inputs"]["workflow"] for x in practice}):
    ids = [str(x["id"]) for x in practice if x["inputs"]["workflow"] == wf]
    rng.shuffle(ids)
    n = len(ids)
    split_ids[wf] = {"fit":ids[:n//2], "dev":ids[n//2:3*n//4], "test":ids[3*n//4:]}
splits = {s:sum([split_ids[w][s] for w in sorted(split_ids)],[]) for s in ["fit","dev","test"]}
json.dump(splits, open(WORK/"analysis_splits.json","w"))
note("# Decision-model weight repair\n"
     "Initial state: no edits intended. Seed 82149, workflow-stratified fit/dev/test split "
     +str({k:len(v) for k,v in splits.items()})+
     ". Test IDs reserved for final paired measurement, not weight fitting.\n"
     "Next forward: original DECIDE, all 900 practice IDs; establish baseline and inspect fit/dev errors. Exact edits: none.")
json.dump({"type":"decide","model":"original","problems":[str(x["id"]) for x in practice]},open(WORK/"baseline_request.json","w"))
```
