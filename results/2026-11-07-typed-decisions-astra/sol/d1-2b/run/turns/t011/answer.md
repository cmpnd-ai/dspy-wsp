I’ll apply a second independent gain pattern. Comparing both probes with the original will let me separate the branch contribution from each output’s unknown RMS scale, then check whether that reconstruction predicts both probes’ logits.

```python
# Hypothesis: two independent interventions identify the branch contribution up to an irrelevant residual scale.
save_edit(gn,g0+d2)
note("Pass 4 READ: causal probe 2, last post_ffn_norm = g0 + second independent random-sign*g0, same 200 fit cases. No other edits.")
```
