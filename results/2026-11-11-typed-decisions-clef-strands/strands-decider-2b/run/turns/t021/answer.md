Full strength also matches the prediction. I’ll test 1.5× strength to check whether pushing the same direction too far causes overcorrection, rather than assuming that larger edits always help.

```python
# Hypothesis: excessive strength will overshoot useful evidence corrections, producing a peaked rather than monotonic accuracy response.
lognote("Pass 7 actual paired dev, strength 1.0: .577778 -> .747778, gain .170000, "
        "paired SE .021344; 103 improved, 32 worsened. Matches cached-head prediction.")
save_rank1(1.5)
lognote("## Before pass 8\nExact edit: k.weight = K0 + 1.5*outer(u,v), lambda .001; all other tensors original. "
        "Same 180 development cases in paired_dev.json. Overshoot dose-response test; "
        "full strength remains provisional best, not yet validation-confirmed.")
print("1.5x predicted dev:",accuracy_stats(dev,dev["L"]+1.5*Cdev))
```
