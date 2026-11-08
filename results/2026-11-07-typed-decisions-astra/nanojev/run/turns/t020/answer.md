The frozen edit gains 19.9 ± 1.6 percentage points on the independent practice test. I’ll now run the required hidden validation comparison before treating it as a confirmed improvement.
```python
# Hypothesis: the scalar readout repair generalizes beyond the visible practice distribution.
note("Forward7 independent practice test300: original0.358000, edited0.556667, paired gain0.198667 SE0.016027; 189 improved,44 worsened. Candidate was fit/selected without these300.")
note("Planned forward8: paired hidden VALIDATION300. Exact current edit scalar.weight=w_original+d_soft (candidate_weights/soft_full_scalar.pt), no other changes. No validation records or labels will be available.")
print("Validation candidate:",edited())
```
