I’ll record the half-strength results and test 1.5× strength. These probes are diagnostic only; they have not been used to fit the correction.
```python
# Hypothesis: stronger readout increases fact sensitivity, but may overshoot decision boundaries.
print("Half strength",cue_stats(20))
save_edit("scalar.weight",torch.tensor(W0.reshape(-1)+1.5*delta900,dtype=torch.float32).reshape(1,-1))
with notes.open("a") as f:
    f.write(f"Pass20 half probe results {cue_stats(20)}.\nPass21 scalar original+1.5*delta900 on same420 probes.\n")
```
