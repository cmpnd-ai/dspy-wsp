Full strength gives +8.00 ± 1.09 points, again matching the reconstruction. I’ll test 1.5× strength next; the hypothesis predicts that overcorrection will reduce the gain rather than keep improving it.

```python
# Hypothesis: excessive readout correction overshoots useful option distinctions.
save_edit("head.scorer.3.weight",torch.tensor(w0+1.5*delta).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass8 full-dose: reserved gain .08000±.01094, .56133→.64133; 208 improved/100 worsened. Current edit 1.5-dose; next same reserved 450, predicted overshoot gain .0258.\n")
```
