I’ll now restrict the correction to high-variance directions in the fitting representations. This tests whether the repair depends on lower-variance information that the original head failed to read.

```python
# Hypothesis: high-variance representation directions alone do not explain the validated gain.
install_job(6)
print("direction norms:", [(name, float(np.linalg.norm(direction))) for name,direction in jobs])
with (work/"notes.md").open("a") as f:
    f.write("\nPass 20: diagnostic covariance eigenvalue >=1 projection of dw600; scalar only; fixed fitting panel + synthetic cues.\n")
```