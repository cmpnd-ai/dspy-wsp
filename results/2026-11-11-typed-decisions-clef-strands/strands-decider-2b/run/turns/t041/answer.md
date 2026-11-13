The higher-rank edit did not improve on the simpler rank-one repair on the reserved practice set. I’ll compare their paired results, restore the validation-confirmed edit, and inspect the remaining errors before deciding whether another change is justified.

```python
# Hypothesis: extra query-context modes add variance without improving the validated rank-one repair.
import os, json, math, shutil, pathlib, collections, inspect
import numpy as np
import torch
WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/strands-2b-astra/work")
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
r13 = read_jsonl(WORK / "gen_13_edited.jsonl")
r15 = read_jsonl(WORK / "gen_15_edited.jsonl")
print("Record fields:", list(r15[0]))
print("Existing globals:")
for name, val in list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "NAMES", "SHAPES"}:
        continue
    if isinstance(val, torch.Tensor):
        print(name, "tensor", tuple(val.shape), val.dtype)
    elif isinstance(val, dict) and len(val) <= 30:
        print(name, "dict", list(val)[:22])
    elif isinstance(val, (list, tuple)) and len(val) >= 100:
        print(name, type(val).__name__, len(val))
    elif inspect.isfunction(val) and val.__module__ == "__main__":
        print(name, "function")
print("Best_2 files:", [str(p.relative_to(WORK / "best_2")) for p in (WORK / "best_2").rglob("*")])
```
