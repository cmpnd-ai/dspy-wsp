The shared-only scorer differs from the cached logits by a possible common offset, which would not affect option rankings. I’ll verify the option contrasts and accuracy before spending a forward pass.
```python
# Hypothesis: the apparent shared-only reconstruction discrepancy is an irrelevant question-common logit offset.
shared_check=logits_scorer_params(D["x"],shared_only_params)
print("Shared-only vs cached maximum option-contrast error:",
      float((option_center(shared_check,D["mask"])-option_center(base_linear,D["mask"])).abs().max()))
print("Shared-only dev metrics:",evaluate_logits(shared_check))
print("Definition direction agreement:",
      float((shared_only_params[0][pair_units[0]]-candidate_dx/candidate_dx.norm()).abs().max()))
note("\nPre-pass21 shared-only audit: raw logits include a centering constant; option-contrast error and dev metrics checked to ensure exact best1 behaviour. Exact installed shared_only_frozen.pt unchanged; next pass21 both228 reserved test IDs.")
```
