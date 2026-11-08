I’ll check whether the stronger violation readout improves the intended agent decisions or starts causing collateral errors, then compare it with the weaker edit.
```python
# Hypothesis: increasing the violation readout from gain 2 to gain 4 may saturate benefits and increase collateral errors.
import os, json, pathlib, numpy as np, torch
WORK = pathlib.Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-astra-strict/work")
print("Kernel variables:", sorted(k for k in globals() if not k.startswith("_")))
print("Analysis helpers:", [k for k,v in list(globals().items()) if callable(v) and getattr(v, "__module__", None) == "__main__"])
print("Pass 11 metadata:", (WORK/"fwd_11"/"meta.json").read_text()[:3500])
print("Files:", sorted(p.name for p in WORK.iterdir()))
```
