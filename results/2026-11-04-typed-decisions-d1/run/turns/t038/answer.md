The prior-only edit yields just +2.04 ± 1.10 points. I’ll now test the complementary context component, which has zero mean effect in the question-option groups used for the decomposition.

```python
# Hypothesis: context-dependent feature variation contributes independently to the repair.
save_edit("head.scorer.3.weight",torch.tensor(w0+context_delta).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass16 prior-only ablation gain .02044±.01104. Current diagnostic edit context-only complementary projection; next both same 450 cases. Best remains full all900 in best_all.\n")
```
