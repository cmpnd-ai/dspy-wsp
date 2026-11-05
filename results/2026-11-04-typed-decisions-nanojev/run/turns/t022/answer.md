The independent practice test improves from 36.2% to 57.47%, a gain of 21.27 ± 1.54 percentage points. I’ll now run the required held-out validation before keeping the edit.

```python
# Hypothesis: the frozen scalar repair transfers beyond the practice distribution used for fitting and selection.
import shutil
shutil.copytree(work/"edited",work/"candidate_linear600",dirs_exist_ok=True)
with open(work/"notes.md","a") as f:
    f.write("\nIndependent practice test (300): original .36200, edited .574667; gain .212667 ± .015428 paired SE, 199 improved / 34 worsened.\n"
            "Pass 8: held-out validation, scalar = original + dw600_linear, no other edits. Candidate snapshot candidate_linear600.\n")
```
