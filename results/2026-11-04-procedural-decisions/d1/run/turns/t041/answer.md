The evidence-only ablation improved accuracy slightly, but neither component alone reproduces the full repair. I’ll record that result and inspect the saved analysis before choosing the next controlled test.

```python
# Hypothesis: the repair requires jointly correcting option bias and reading record-sensitive features.
from pathlib import Path
import json, torch, numpy as np
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/procedural-decisions/d1/work")
with (work/"notes.md").open("a") as f:
    f.write("\nPass 16 evidence-only complement .54603 vs .51984, gain .02619 SE .00607; 32 improved/5 worsened. Full C .80635 on same explored cases; both subspaces are needed. Current edit is diagnostic ablation, NOT final.\n")
print([(k, tuple(v.shape)) for k,v in list(globals().items()) if isinstance(v,(torch.Tensor,np.ndarray))])
print("Files:", [p.name for p in work.iterdir() if p.is_file()])
```
