Keeping only the largest 512 coefficient changes loses almost all the benefit. That suggests coefficient magnitude alone does not identify the functional repair: smaller terms may be needed to balance correlated neuron activations. I’ll test the complementary half.

```python
# Hypothesis: small readout terms jointly balance the larger corrections; neither half alone reproduces the repair.
complement_delta=delta_all-sparse_delta
save_edit("head.scorer.3.weight",load_original("head.scorer.3.weight")+torch.tensor(complement_delta,dtype=torch.float32).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass22 top512 by absolute coefficient correction: diagnostic450 +.00444±.01303 (not the earlier activation-weighted ranking). Current complementary512 correction; next same450. Evidence favors distributed balanced readout, not a few largest coefficients.\n")
```
