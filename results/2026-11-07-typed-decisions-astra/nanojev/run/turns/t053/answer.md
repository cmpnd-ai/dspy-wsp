The decomposed candidate did not improve the validation score. I’ll now test the second candidate already selected from practice cross-fitting: strength 0.875 along the full correction direction. After that, I’ll stop selecting weights and use the remaining passes for diagnostics.
```python
# Hypothesis: a modest increase along the evidence-sensitive repair direction preserves the validation gain without excessive amplification.
note("\nForward18 decomposed candidate hidden validation300: original .348000, edited .570667, paired gain .222667 SE .014754;206 improved/31 worsened. Lower score than best3(.579333); not retained.")
save_edit("scalar.weight",w0875_900)
note("\nPlanned forward19: paired hidden VALIDATION300. Exact edit scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt; all other tensors original. Candidate B was fixed before forward18. After this, freeze best validation-confirmed candidate; remaining passes are practice verification, matched-cue diagnostics and a prior-only ablation, not further recipe selection.")
print("Current edit:",edited(),"correction norm",float(torch.linalg.vector_norm(load("scalar.weight")-load_original("scalar.weight"))))
```
