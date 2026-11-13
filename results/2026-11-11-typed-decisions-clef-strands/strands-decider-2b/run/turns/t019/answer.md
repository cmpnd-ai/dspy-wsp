The actual half-strength edit matches the prediction: a 14.56-point gain, with paired SE 1.64 points. I’ll test full strength next, keeping the direction fixed so this is a clean dose-response experiment.

```python
# Hypothesis: increasing the same evidence-readout direction from half to full strength will produce the predicted additional gain.
lognote("Pass 6 actual paired dev, strength .5: original .577778 -> edited .723333, "
        "gain .145556, paired SE .016363; 98 improved, 20 worsened. Matches cached-head prediction.")
save_rank1(1.0)
lognote("## Before pass 7\nExact edit: k.weight = K0 + 1.0*outer(u,v), lambda .001, all other tensors original. "
        "Same 180 dev IDs in paired_dev.json. Direction unchanged; full-strength causal test.")
print("Full-strength predicted dev:",accuracy_stats(dev,dev["L"]+Cdev))
```
