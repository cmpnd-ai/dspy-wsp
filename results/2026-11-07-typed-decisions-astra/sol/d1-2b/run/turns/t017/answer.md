The half-strength edit gains 3.93 percentage points (paired SE 0.55) on the fit split. I’ll now double the strength to test the predicted dose-response, then compare which questions changed.

```python
# Hypothesis: increasing the same localized correction produces a predictable additional improvement.
save_edit(gn,g0*(1+torch.tensor(v,dtype=torch.float32)))
note("Pass 5 half-strength: fit gain +.039286 SE .005523 (103 improved,28 worsened). Next pass 6: strength 1.0, same ridge=1 384-dim direction, same 448 fit cases; dose-response.")
```
