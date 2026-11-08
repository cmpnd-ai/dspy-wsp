Strength two gives an 8.04-point fit gain (SE 1.14), though regressions also increase. I’ll measure half strength and strength three to check that this is a bounded response, not an assumption that larger changes always help.

```python
# Hypothesis: candidate C has a bounded dose-response, with half strength helping less than strengths one and two.
save_edit(gn,g0*(1+torch.tensor(.5*v3,dtype=torch.float32)))
note("Pass14 C strength2 fit +.080357 SE .011409. Pass15 C strength.5 same448 fit, lower-dose bracket.")
```
