The restored edit improves the fitting panel by 28.49 percentage points. I’ll use the last pass to confirm that the exact submitted weights still reproduce the held-out validation gain.

```python
# Hypothesis: restoring best_1 exactly reproduces its previously confirmed validation result.
assert torch.equal(load("scalar.weight"),torch.from_numpy(np.load(work/"best_1"/"scalar.weight.npy")))
print("Current edits:", edited())
with (work/"notes.md").open("a") as f:
    f.write("\nPass 24 fitting-panel result: .363111 -> .648000, paired gain .284889 ± .012895 (450 cases; descriptive only).\nPass 25: final held-out validation confirmation. Exact best_1 scalar.weight restored; all other weights original. No further selection or fitting.\n")
```