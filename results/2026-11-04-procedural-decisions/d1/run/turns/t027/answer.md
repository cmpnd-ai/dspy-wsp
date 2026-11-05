Candidate C’s actual reserve result matches the replay. I’ll log the uncertainty and test it on validation; the reserve result alone is not strong enough to establish a gain.

```python
# Hypothesis: pooling more observations improves generalization, which must be checked independently on validation.
with open(work/"notes.md","a") as f:
    f.write("Pass 9 C reserve actual .52778 -> .63333, gain .10556 SE .06162, too small a sample for confidence. Next validation C, unchanged final readout.\n")
```
