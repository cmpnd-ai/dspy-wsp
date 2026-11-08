The half-strength run matched the scorer-only prediction exactly. I’ll now double the conditional extension, keeping the shared residual fixed; this tests whether stronger cue readout eventually harms ranking rather than simply improving it monotonically.
```python
# Hypothesis: doubling the conditional extension will overcorrect some questions, reducing accuracy relative to the validated dose.
save_scorer_params(double_cond_params)
note("\nPass16 actual half conditional extension: dev55.000% ->69.375%, gain14.375pp SE1.696pp,130 improved/33 worsened; exact local prediction.\nNext pass17: DECIDE both224 dev IDs in dose_dev.json. Exact edit conditional_double.pt: same base pair33/746 at1x, same54 conditional channel input directions/gates/biases; conditional final readout entries +/-40 (2x), all other weights original or as best2. Expected "+str(evaluate_logits(double_cond_logits))+". best_2 backed up; diagnostic intervention only.")
print("Installed conditional dose 2x; expected",evaluate_logits(double_cond_logits))
```
