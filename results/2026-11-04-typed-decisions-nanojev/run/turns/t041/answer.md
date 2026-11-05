I’ll inspect the surviving variables and diagnostic requests before restoring the validated edit.

```python
# Hypothesis: retain the validated scalar repair rather than the temporary dose ablation.
from pathlib import Path
import json, numpy as np, torch
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/nanojev/work")
print([(k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "")
       for k, v in list(globals().items())
       if not k.startswith("_") and k not in ["NAMES", "SHAPES"]][-100:])
print("Snapshots:", [p.name for p in work.iterdir() if p.is_dir()])
print("Read metadata:", (work/"fwd_16"/"meta.json").read_text()[:2600])
```