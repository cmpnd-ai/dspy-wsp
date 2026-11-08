The updated strength-one edit gains 5.45 points (SE 0.76) on fit cases. I’ll test strength two to confirm the larger dataset’s predicted dose-response.

```python
# Hypothesis: candidate B improves further at strength two before nonlinear saturation.
save_edit(gn,g0*(1+torch.tensor(2*v2,dtype=torch.float32)))
note("Pass11 candidate B strength1 fit +.054464 SE .007568. Pass12 candidate B strength2 on same448 fit.")
```
