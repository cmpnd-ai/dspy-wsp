The context-only component retains a validation gain of 2.67 ± 0.99 percentage points, while the full correction gains 8.07 ± 1.26. Both components together work better than either alone. I’ll use two final neuron-subset ablations to check whether the repair is distributed across the scorer’s features.

```python
# Hypothesis: the correction is distributed across scorer GELU features, not a single global scale or small neuron switch.
ix=np.argsort(np.abs(delta_all))[-512:]
sparse_delta=np.zeros_like(delta_all); sparse_delta[ix]=delta_all[ix]
save_edit("head.scorer.3.weight",load_original("head.scorer.3.weight")+torch.tensor(sparse_delta,dtype=torch.float32).reshape(1,-1))
origw=load_original("head.scorer.3.weight").numpy().ravel()
print("Correction cosine to original readout:",np.dot(origw,delta_all)/(np.linalg.norm(origw)*np.linalg.norm(delta_all)))
print("Largest correction neuron indices:",np.argsort(np.abs(delta_all))[-12:][::-1].tolist())
with (work/"notes.md").open("a") as f:
    f.write("\nPass21 context-only VALIDATION +.02667±.00987; full combination best +.08067±.01260. Current top512 magnitude entries of all900 correction only; next diagnostic same450 practice. Then complementary512 ablation, full-edit read, final validation.\n")
```
