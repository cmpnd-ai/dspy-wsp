The state-dependence ablation lowered accuracy by 3.39 percentage points, as predicted. I’ll restore the validated full repair and freeze its design, then measure it on the 228 practice examples reserved from both fitting and development selection.
```python
# Hypothesis: the validated scorer repair generalizes to practice examples excluded from both fitting and candidate selection.
save_scorer_params(best2_params)
assert all(torch.equal(load(name),val) for name,val in zip(
    ["head.scorer.1.weight","head.scorer.1.bias","head.scorer.3.weight"],best2_params))
test_request={"type":"decide","model":"both","problems":splits["test"]}
json.dump(test_request,open(W/"frozen_test.json","w"))
note("\nPass18 actual conditional state-direction ablation: dev55.000% ->66.696%, +11.696pp SE1.703pp. Full conditional repair exceeds prior-only by3.393pp, paired SE1.003pp on224dev items. Confirms conditional gain requires state-dependent directions, not only answer priors.\nDESIGN FROZEN: restore best2 conditional_candidate.pt exactly; no further fitting/selection using test examples. Invoice-specific attempted extensions rejected for negligible dev benefit. All remaining interventions diagnostic only unless explicitly noted.\nNext pass19: DECIDE both on228 untouched test practice IDs in frozen_test.json. Exact edit best2: scorer.1 rows/biases33/746 shared residual and108 low-contribution q/option gated rows; scorer.3 corresponding110 entries. Entirely identical to validation pass13. Test IDs were excluded from weight fitting and hyperparameter selection.")
print("Restored best2. Reserved test count",len(splits["test"]))
```
