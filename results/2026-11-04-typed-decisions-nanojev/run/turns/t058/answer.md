I’ll lower the cutoff once more to determine whether nearly all of the correction is needed.

```python
# Hypothesis: retaining directions down to eigenvalue 0.01 nearly reproduces the full correction.
install_job(8)
with (work/"notes.md").open("a") as f:
    f.write("\nPass 22: diagnostic covariance eigenvalue >=0.01 projection of dw600, same fixed panel and cues.\n")
```