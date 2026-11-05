The repair shows a dose-response in head-only calculations: internal holdout accuracy rises from 36.3% at zero strength to 53.1% at 0.75 strength, then declines when over-applied. I’ll now intervene on the actual scalar weights at 0.25 strength to verify that the full model behaves as predicted.

```python
# Hypothesis: directly strengthening the identified residual-logit direction will causally reverse collapsed candidate preferences.
chosen_lambda=.03
dw400=candidate_dw[chosen_lambda].copy()
strength=.25
save_edit("scalar.weight",torch.from_numpy((w0+strength*dw400).astype(np.float32))[None,:])
internal_ids=fit_chunks[2]
json.dump({"type":"decide","model":"both","problems":internal_ids},open(work/"dose_request.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nClosed-form scalar correction from first 400 fitting cases: centered log gold probabilities minus original logits, ridge lambda .03. No gradients. Direction norm 61.72 versus original head norm .551.\n"
            "Internal 200-case head-only dose response: strength 0/.25/.5/.75/1/1.25 => accuracy .363/.429/.509/.531/.524/.511. Set corrections typical within-choice range .142 logits.\n"
            "Pass 5: actual scalar.weight = original + .25 * dw400; all other tensors original; paired decide on internal 200 fitting cases.\n")
```
