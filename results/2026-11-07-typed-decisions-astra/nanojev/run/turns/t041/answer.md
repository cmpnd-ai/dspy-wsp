The scalar-head repair is already confirmed on validation. I’ll first recover the cached analysis and fitting recipe, then use the newly completed feature cache to test whether a fixed-recipe refit improves it without changing the backbone.
```python
# Hypothesis: the remaining gains can be tested with a fixed closed-form readout recipe, using cached features only.
import os, json, inspect
from pathlib import Path
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/nanojev-astra/work")
print("Edits:", edited())
print("Kernel functions:", [(k, str(inspect.signature(v))) for k,v in list(globals().items())
                            if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"][:100])
print("Important cached objects:")
for k,v in list(globals().items()):
    if k.startswith("_") or k in ("NAMES", "SHAPES"):
        continue
    if hasattr(v,"shape") and not inspect.ismodule(v):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list,dict)) and any(s in k.lower() for s in ["row","split","record","data","cache","fit","dev","test","meta","head"]):
        print(k, type(v).__name__, len(v))
print("Candidate files:", sorted(p.name for p in (work/"candidate_weights").iterdir()))
print("Analysis files:", sorted(p.name for p in work.iterdir() if p.suffix in [".py", ".json", ".md"]))
```
