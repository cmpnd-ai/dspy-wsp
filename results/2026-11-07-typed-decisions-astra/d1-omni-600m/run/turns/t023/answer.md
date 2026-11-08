Full strength also matches the prediction: +12.05 points on development, paired SE 1.68. I’ll now check the untouched validation set before treating this as a keepable repair.
```python
# Hypothesis: the scorer-channel repair generalizes beyond the fit/development examples used to derive and select it.
note("Pass 8 actual full-strength intervention: original 55.000%, edited 67.054%, +12.054pp, paired SE 1.678pp; 119 improved / 39 worsened. Matches predicted scorer-only effect exactly, confirming causal edit location. Quarter->full dose gain grows as predicted.\n"
     "Next pass: VALIDATION both on all 300 hidden examples. Exact edits remain candidate_linear_full: scorer.1 rows/biases 33/746 +/- residual direction and centering bias, scorer.3 entries +/- ||direction||, strength 1.0. No validation examples have been observed or used in fitting.")
```
This validation run tests the frozen full-strength candidate on 300 hidden examples. I’ll retain it only if the paired gain is clearly larger than its uncertainty.
```forward
{"type":"decide","model":"both","validation":true}
```
The bookkeeping block should run first; the validation call will follow in the next turn.
