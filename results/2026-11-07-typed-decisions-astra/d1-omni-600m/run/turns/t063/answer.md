On reserved examples, the gates activate for the intended question/option 99.96% of the time, with a 0.004% false-positive rate. The displaced original neurons contribute only 0.00050 RMS to logit contrasts. I’ll now measure the shared-only repair on the full reserved set to isolate the conditional extension’s independent contribution.
```python
# Hypothesis: the conditional channels add generalizable signal beyond the validated shared residual.
shared_only_params=tuple(scorer_params[k].clone() for k in ["1.weight","1.bias","3.weight"])
for u in pair_units:
    shared_only_params[0][u]=best2_params[0][u]
    shared_only_params[1][u]=best2_params[1][u]
    shared_only_params[2][0,u]=best2_params[2][0,u]
print("Shared-only dev reconstruction error",float((logits_scorer_params(D["x"],shared_only_params)-base_linear).abs().max()))
save_scorer_params(shared_only_params)
torch.save({"params":shared_only_params,"purpose":"frozen shared-only ablation"},W/"shared_only_frozen.pt")
note("\nPass20 audit: reserved-test q/option gates TPR99.9646%, FPR0.00421%; displaced original-neuron option-contrast RMS0.000496. Local full scorer reproduces actual edited probabilities max error0.000428.\nNext pass21: DECIDE both all228 reserved test IDs in frozen_test.json. Diagnostic shared-only ablation, exact shared_only_frozen.pt: scorer.1 rows/biases33/746 and scorer.3 entries33/746 as validated best1 atstrength1; other neurons original. No new fitting or dose selection. Compare with frozen best2 pass19 to measure independent contribution of conditional channels. Final choice remains validation-confirmed best2.")
```
