I’ll first map the practice schemas and establish the model’s baseline errors. That will let me target a specific decision failure rather than change confidence globally.

This inspection will check label balance and available head code, and create the experiment log before any forward pass.


```python
# Hypothesis: baseline errors will cluster by field and option, exposing a repairable head-level bias or missing cue.
import os, json, math, time, pathlib, collections, random
import numpy as np
import torch
import torch.nn.functional as F

WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/clef-flash-astra/work")
MODEL = pathlib.Path("/Users/dbreunig/.cache/huggingface/hub/models--Cloudflare--clef-flash/snapshots/22b75507981345858dd152fbd9a6f8c0c4068b0a")
data = [json.loads(s) for s in (WORK/"practice/trainset.jsonl").read_text().splitlines() if s.strip()]
byid = {str(d["id"]): d for d in data}
notes_path = WORK/"notes.md"
def note(text):
    with notes_path.open("a") as f:
        f.write(text.rstrip()+"\n\n")
note("# Head weight repair experiment\nBackbone fixed. No gradients, optimizers, or iterative weight fitting. All proposed edits will be closed-form or targeted causal interventions.")
print("Examples:", len(data), "workflows:", dict(collections.Counter(d["inputs"]["workflow"] for d in data)))
print("Current edits:", edited())
print("Example:", json.dumps(data[0], ensure_ascii=False)[:6500])
print("Model files:", [p.name for p in MODEL.iterdir()])
print("Head tensor count:", len([n for n in NAMES if not n.startswith("model.")]))
baseline_ids = [str(d["id"]) for d in data]
json.dump({"type":"decide","model":"original","problems":baseline_ids}, (WORK/"baseline_request.json").open("w"))
note("## Planned pass 1 — baseline\nExact edit: none. DECIDE original on all 900 practice IDs, stored in baseline_request.json. Will diagnose errors by workflow, field, gold/predicted options, and confidence.")
```
