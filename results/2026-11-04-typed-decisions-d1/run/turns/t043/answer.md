Half strength improves accuracy, but less than full strength. I’ll now test 1.5 strength to check for overshoot, then inspect which question types and cue responses the correction changes.

```python
# Hypothesis: excessive readout correction overshoots useful option margins.
import numpy as np, torch, json
save_edit("head.scorer.3.weight", load_original("head.scorer.3.weight") + torch.as_tensor(1.5 * delta_all, dtype=torch.float32).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass18 half-dose all900: +.05289±.00752 on diagnostic 450. Current edit 1.5*all900 delta; next same450 paired dose test. Best remains validated full all900.\n")
print("Variables:", [(k, type(v).__name__, getattr(v,"shape",None)) for k,v in list(globals().items()) if k in ["features","meta","rows","records","examples","delta_all","prior_delta","context_delta"]])
print("Functions:", [k for k,v in list(globals().items()) if callable(v) and getattr(v,"__module__",None)=="__main__"])
print("Cue meta:", (work/"fwd_11"/"meta.json").read_text()[:2400])
```
