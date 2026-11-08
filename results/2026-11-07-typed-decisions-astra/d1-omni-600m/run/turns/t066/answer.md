The shared-only repair gained 12.46 points on the reserved set, versus 14.04 points for the full repair. I’ll measure their paired difference, then test the prior-only conditional ablation on the same reserved examples to check whether its state-dependent contribution generalizes.
```python
# Hypothesis: the conditional extension's independent benefit comes from state-dependent evidence rather than fixed option preferences.
shared_test={r["id"]:r for r in map(json.loads,open(W/"gen_21_edited.jsonl"))}
test_extension=np.array([te[i]["score"]-shared_test[i]["score"] for i in splits["test"]])
print("Full minus shared-only, reserved test:",
      {"gain":float(test_extension.mean()),"se":float(test_extension.std(ddof=1)/len(test_extension)**.5),
       "improved":int((test_extension>0).sum()),"worsened":int((test_extension<0).sum())})
save_scorer_params(prior_only_params)
note("\nPass21 shared-only reserved test:58.421% ->70.877%, +12.456pp paired SE1.596pp. Full best2 independent increment over shared="+str(float(test_extension.mean()))+" SE="+str(float(test_extension.std(ddof=1)/len(test_extension)**.5))+".\nNext pass22: DECIDE both228 reserved test IDs in frozen_test.json. Exact diagnostic edit conditional_prior_only.pt: shared pair33/746 at1x,54 conditional gates retained but state-dependent vectors removed and replaced by per-q/option fit mean corrections. Identical ablation to pass18, no test refitting. Final submitted choice remains best2, the largest validation-confirmed gain.")
```
