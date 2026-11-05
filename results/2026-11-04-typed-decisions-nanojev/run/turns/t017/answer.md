The actual intervention exactly matches the head-only prediction: a gain of 6.6 percentage points, with paired SE 1.15 points. I’ll increase the same direction to 0.75 strength to test the predicted larger effect.

```python
# Hypothesis: increasing the same scalar direction to .75 will raise internal accuracy to approximately .531.
save_edit("scalar.weight",torch.from_numpy((w0+.75*dw400).astype(np.float32))[None,:])
with open(work/"notes.md","a") as f:
    f.write("\nPass 5 confirmed causal prediction exactly: .363 -> .429, paired gain .066 ± .01153, 72 improved / 22 worsened.\n"
            "Pass 6: scalar original + .75*dw400, same internal 200 fitting cases, all other weights unchanged.\n")
```
