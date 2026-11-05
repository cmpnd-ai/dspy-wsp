The gains are concentrated in graded judgments such as risk, outcome, churn risk, and severity; already-strong category and duplicate detection barely change. The cue tests are mixed, so I won’t claim a universal cue-sensitivity repair. To test the narrower explanation—better contextual readout—I’ll evaluate the prior-only and context-only ablations on validation.

```python
# Hypothesis: fixed question/option preferences explain only part of the gain.
save_edit("head.scorer.3.weight", load_original("head.scorer.3.weight") + torch.tensor(prior_delta,dtype=torch.float32).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nDiagnostics: all900 practice largest gains trace outcome +.209, risk +.200; customer churn +.191; invoice urgency +.178; security severity +.173. Cue reads have identical hidden states but mixed cue improvements: invoice overdue urgency sensitivity increases; trace review sensitivity not consistently improved. No universal cue-repair claim. Current prior-only projection of all900 correction; next validation causal ablation (not candidate selection).\n")
```
