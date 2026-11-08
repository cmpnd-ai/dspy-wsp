I’ll repeat the even-coordinate measurement on development records. This gives an independent check of the same causal decomposition before using it to predict candidate edits.
```python
# Hypothesis: the last-write decomposition is stable on records not used to estimate the repair.
note("Pass 6 completed: even-gain probe fit_b, fwd_6.\n\nPlanned pass 7: READ edited dev, 180 IDs in probe_even_dev.json. Exact edit remains only layers.23.post_ffn_norm even coordinates x1.5. Development data will evaluate, not fit, the repair.")
print("Current edit:",edited(),"next request:",str(WORK/"probe_even_dev.json"))
```
