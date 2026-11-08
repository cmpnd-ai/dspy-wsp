Validation confirms the gain: **+7.27 percentage points (paired SE 1.31 points)**. I’ll keep this edit fixed. The remaining passes will audit how it changes decisions, rather than tune it further.

```python
# Hypothesis: the validated gain comes from option-specific residual routing, not uniform sharpening.
import json, numpy as np
with (work/"notes.md").open("a") as f:
    f.write("Pass18 VALIDATION confirms locked edit: original .606667, edited .679333, gain +.072667 SE .013075;123 improved,58 worsened/300. KEEP candidate C strength2. Remaining passes diagnostic only; no further selection.\n")
print("Files:", [p.name for p in work.iterdir() if p.is_file()])
print("Arrays:", [(k, v.shape) for k,v in list(globals().items()) if isinstance(v,np.ndarray)])
r17o=[json.loads(s) for s in (work/"gen_17_original.jsonl").read_text().splitlines()]
r17e=[json.loads(s) for s in (work/"gen_17_edited.jsonl").read_text().splitlines()]
print("Record keys:",r17o[0].keys())
print(str(r17o[0])[:6500])
print("Read metadata:",(work/"fwd_2"/"meta.json").read_text()[:4500])
```
