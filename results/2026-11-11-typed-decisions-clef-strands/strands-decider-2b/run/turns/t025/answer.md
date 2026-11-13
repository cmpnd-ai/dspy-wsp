The reserved practice test confirms a 13.56-point gain, with paired SE 2.19 points. I’ll now run the required hidden validation check before treating this edit as a keeper.

```python
# Hypothesis: the state-dependent readout repair generalizes beyond the exposed practice set.
lognote("Pass 9 reserved practice-test, strength 1.0: original .591111 -> edited .726667, "
        "gain .135556, paired SE .021871; 98 improved, 37 worsened. "
        "Candidate fixed before test; gain exceeds six paired SE.")
lognote("## Before pass 10\nExact edit: k.weight = K0 + outer(u,v), lambda .001; all other weights original. "
        "Hidden VALIDATION both on all 300 cases; no validation examples or per-item records are visible.")
print("Current edits:",edited())
```
