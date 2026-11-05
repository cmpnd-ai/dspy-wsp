I’ll now check 1.5× strength on the standard 450-item practice cohort. This cohort is part of the expanded fit, so the result is a dose-response diagnostic, not independent generalization evidence.
```python
# Hypothesis: overshooting the corrected readout can increase cue sensitivity while reducing overall accuracy.
print("1.5 strength probes",cue_stats(21))
with notes.open("a") as f:
    f.write(f"Pass21 1.5x probes {cue_stats(21)}.\nPass22: scalar original+1.5*delta900, both on test450 (now in-sample). Diagnostic strength sweep only; validation-confirmed full direction still incumbent.\n")
json.dump({"type":"decide","model":"both","problems":test450},(p/"dose_decide.json").open("w"))
```
