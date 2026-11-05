At 1.5× strength, the gain falls to +2.58 ± 1.29 points, as predicted. I’ll restore full strength and validate it on the unseen 300-case set before retaining it.

```python
# Hypothesis: the full-dose readout repair generalizes beyond all practice-based decisions.
save_edit("head.scorer.3.weight",torch.tensor(w0+delta).reshape(1,-1))
import shutil
if (work/"candidate_full").exists():shutil.rmtree(work/"candidate_full")
shutil.copytree(work/"edited",work/"candidate_full")
with notes.open("a") as f:
    f.write("\nPass9 1.5-dose: gain .02578±.01293, confirming overshoot. Restored full-dose ridge1 scorer readout. Next validation both, all 300 unseen cases; no validation labels used in fitting.\n")
```
