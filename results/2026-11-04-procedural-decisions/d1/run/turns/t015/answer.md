The full program exactly reproduces the replayed gain. I’ll confirm candidate A on held-out validation before treating it as a viable final edit.

```python
# Hypothesis: the scorer repair generalizes beyond both the fitting and practice-test examples.
with open(work/"notes.md","a") as f:
    f.write("Pass 4 actual paired practice confirms exactly: .51984 -> .69365, +.17381 SE .02206, 183 improved/68 worsened. Next validation candidate A, unchanged head.scorer.3.weight; no other edits.\n")
```
