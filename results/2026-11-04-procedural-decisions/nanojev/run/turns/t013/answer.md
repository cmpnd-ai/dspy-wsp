Half-strength delivers the predicted improvement: +5.19 percentage points, paired SE 0.94 points. I’ll now test the full evidence correction; unlike a temperature change, this should reverse some boolean decisions and more strongly repair severity rankings.

```python
# Hypothesis: full strength crosses evidence-dependent decision boundaries that half strength cannot reach.
save_edit("scalar.weight",torch.from_numpy((W0+delta).astype(np.float32)[None]))
with open(notes,"a") as f:
    f.write("\nPass 3 half-strength, disjoint 450: .47185 -> .52370, gain .05185 ± .00935 paired SE (76 improved,25 worsened)."
            "\nPass 4: scalar.weight original + 1.0*delta, same 450 for dose-response; no other edits.\n")
```
