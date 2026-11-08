The reserved practice test improves from 60.78% to 76.44%: a gain of **15.67 ± 1.33 percentage points** (paired SE), with 208 records improved and 49 worsened. I’ll now check the hidden validation set before treating this as a confirmed candidate.
```python
# Hypothesis: the independently measured practice gain carries over to the hidden validation distribution.
note("Pass 14 independent practice-test result, n=360: original .607778, edited .764444; paired gain +.156667, SE .013341; 208 improved, 49 worsened. This is ~11.7 paired SE above zero. Candidate: rank9 ridge1 alpha1 W2 repair.\n\nPlanned pass 15: DECIDE both on hidden validation, n=300. Exact current edit: layers.23.feed_forward.w2.weight = W2_original + B A_ridge1 W2_original, alpha1; only this tensor changed. Validation has not yet been queried.")
print("Current edits:",edited())
```
