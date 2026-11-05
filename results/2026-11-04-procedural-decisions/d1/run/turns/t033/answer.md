The 90% dose reaches 71.3%, as predicted. I’ll restore the full candidate C and measure its full-dose behavior on the same explored practice cases; these cases are now part of its fit, so this is a causal check, not an independent generalization estimate.

```python
# Hypothesis: full correction removes the remaining option bias and restores all impact levels.
save_edit("head.scorer.3.weight",candidateC)
with open(work/"notes.md","a") as f:
    f.write("Pass 12 C 90% dose .71270, gain .19286 SE .01908 on explored test. Next full C on same 420 (included in C fit; causal, not generalization).\n")
```
