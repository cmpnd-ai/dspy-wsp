I’ll include the next band of representation directions, lowering the covariance-eigenvalue cutoff from 1 to 0.1.

```python
# Hypothesis: adding lower-variance directions recovers more of the repair's effect.
install_job(7)
with (work/"notes.md").open("a") as f:
    f.write("\nPass 21: diagnostic covariance eigenvalue >=0.1 projection of dw600; same fixed panel; not a candidate-selection run.\n")
```