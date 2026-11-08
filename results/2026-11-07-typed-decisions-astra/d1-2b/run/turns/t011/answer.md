I’ll also read the 180 development records now, before any intervention. Their labels will select the repair strength, while the reserved 360-record practice test and hidden validation set will remain checks on generalization.
```python
# Hypothesis: a useful repair should generalize to untouched development states, not just interpolate the fitting records.
note("Pass 3 completed: original fit_b read, 900 questions in fwd_3.\n\nPlanned pass 4: READ original dev (180 records, 900 questions), read_dev.json. Current edit: none. Fit remains 360 records; dev is for closed-form regularization/strength selection, not coefficient estimation; test=360 reserved for paired confirmation.")
print("Next:", str(WORK/"read_dev.json"), "edit:",edited())
```
