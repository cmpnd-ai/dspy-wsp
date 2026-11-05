The prior-only component gives essentially no validation gain: +0.07 ± 1.26 percentage points. I’ll now test its complementary component, which excludes the mean question-and-option feature directions used by that ablation.

```python
# Hypothesis: context-dependent feature directions retain substantial held-out benefit after removing prior directions.
save_edit("head.scorer.3.weight",load_original("head.scorer.3.weight")+torch.tensor(context_delta,dtype=torch.float32).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass20 prior-only VALIDATION +.00067±.01262, effectively no benefit. Current context-only complement; next validation ablation. Full all900 remains selected best.\n")
```
