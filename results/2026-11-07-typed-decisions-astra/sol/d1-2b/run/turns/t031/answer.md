Candidate B reaches a 6.96-point fit gain at strength two. I’ll also test the less-regularized closed-form solution, which predicted a higher fit score, before making a final selection for holdout testing.

```python
# Hypothesis: lower regularization captures additional useful option-contrast directions without changing the edit's scope.
v3=sol2[.1]
np.save(work/"gain_direction_400_ridge01.npy",v3)
save_edit(gn,g0*(1+torch.tensor(v3,dtype=torch.float32)))
note("Pass12 B strength2 fit +.069643 SE .010344. Next pass13 candidate C ridge.1, same400 fit causal measurements, same384dims, strength1; DECIDE both448 fit.")
```
