The full program matches the predicted half-strength gain exactly: +4.62 ± 0.72 percentage points. I’ll increase the same direction to full strength to test the predicted dose-response.

```python
# Hypothesis: full-strength correction should recover more rankings than the half-strength intervention.
save_edit("head.scorer.3.weight",torch.tensor(w0+delta).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass7 half-dose causal test: reserved 450 original .56133, edited .60756, gain .04622±.00720; 135 improved/59 worsened. Exact local prediction confirmed. Current edit full-dose ridge1 readout correction. Next pass both on same 450.\n")
```
