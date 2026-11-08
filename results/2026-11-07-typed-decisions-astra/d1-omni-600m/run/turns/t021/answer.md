The quarter-strength edit matches the predicted gain exactly: +4.82 percentage points, paired SE 1.05. I’ll increase the same direction to full strength to test the next point on the dose-response curve.
```python
# Hypothesis: increasing only the identified residual channel should produce the predicted larger gain, confirming causal localization to the scorer.
note("Pass 7 actual quarter-strength intervention: original 55.000%, edited 59.821%, gain +4.821pp, paired SE 1.055pp; 58 improved / 21 worsened of 224 dev examples. Exactly matches local scorer prediction.")
w1edit,b1edit,w3edit=install_channel(candidate_dx,1.0,"candidate_full")
import shutil
cand_dir=WORK/"candidate_linear_full"
cand_dir.mkdir(exist_ok=True)
for n in ["head.scorer.1.weight","head.scorer.1.bias","head.scorer.3.weight"]:
    torch.save(load(n),cand_dir/(n+".pt"))
note("Next pass: DECIDE both on the same 224 dev IDs in dose_dev.json. Exact edits: same +/- rows/biases at neurons 33/746; final readout entries now +/- ||candidate_dx|| (strength 1.0), other values original. Definition candidate_full_definition.pt; copies candidate_linear_full/. Preregistered expected accuracy 67.054%, +12.054pp, paired SE 1.678pp.")
```
The full-strength candidate is ready for the paired development check.
