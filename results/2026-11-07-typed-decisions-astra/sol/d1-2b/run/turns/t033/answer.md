Candidate C improves fit accuracy by 7.23 points (SE 0.84), with fewer worsened records than candidate B at strength two. I’ll test C at strength two, then bracket its response with weaker and stronger doses.

```python
# Hypothesis: candidate C's benefit rises toward strength two, as the causal head reconstruction predicts.
save_edit(gn,g0*(1+torch.tensor(2*v3,dtype=torch.float32)))
note("Pass13 C strength1 fit +.072321 SE .008402. Pass14 C strength2 same448 fit.")
```
