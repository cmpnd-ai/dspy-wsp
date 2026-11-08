The locked edit improved the untouched practice holdout by **6.53 percentage points (paired SE 1.14 points)**. I’ll preserve that candidate and document it before testing the held-out validation set.

```python
# Hypothesis: the fit-selected final-FFN gain correction generalizes beyond the practice holdout.
from pathlib import Path
import shutil
work = Path("/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-2b/work")
print((work / "notes.md").read_text())
print("Current edits:", edited())
backup = work / "locked_C_strength2"
if not backup.exists():
    shutil.copytree(work / "edited", backup)
with (work / "notes.md").open("a") as f:
    f.write("\nPass17 independent practice holdout: original .619111, edited .684444, gain +.065333 SE .011405; 192 improved,107 worsened (450 items). Candidate remains LOCKED; no holdout-based selection.\n"
            "Next pass18 VALIDATION both, all300 hidden examples. Exact edit: layers.23.post_ffn_norm.weight = original + 2*candidate-C direction (ridge .1,384 dimensions,400 fit examples); all other tensors unchanged. Backup locked_C_strength2.\n")
```
