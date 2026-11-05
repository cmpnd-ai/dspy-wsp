The ablation behaves as predicted. High-variance directions retain substantial gains, but lose roughly 12 practice points relative to the full fit. Since low-variance directions can also overfit, I’ll check whether the simpler high-only edit generalizes better on validation before restoring the full one.

```python
# Hypothesis: the high-variance-only correction may trade fitting accuracy for better held-out generalization.
with open(notes,"a") as f:
    f.write("\nPass 14 high-only ablation on 450: .47185 -> .68370, gain .21185 ± .02259; confirms loss from removing low-variance directions."
            "\nPass 15 validation: same high-only scalar correction (eigenvalue>=1); full validated edit remains backed up.\n")
```
