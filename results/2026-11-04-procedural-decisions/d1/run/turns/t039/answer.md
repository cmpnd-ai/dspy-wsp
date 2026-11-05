The bias-only intervention loses 10.3 percentage points, matching the replay. I’ll test the complementary evidence-only direction next; it should yield only a small gain while the original option biases remain.

```python
# Hypothesis: original option biases continue to mask evidence unless both readout components are repaired.
save_edit("head.scorer.3.weight",torch.from_numpy(w0+evidenceC).reshape(1,-1))
with open(work/"notes.md","a") as f:
    f.write("Pass 15 bias-only .41667, gain -.10317 SE .01707. Next evidence-only complement, same 420 explored practice cases; predicted .54603.\n")
```
