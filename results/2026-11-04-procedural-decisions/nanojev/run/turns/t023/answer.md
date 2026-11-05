Cross-fold accuracy favors ridge 0.1 at 69.07%; intent remains difficult, but sensors reach 83% and urgency improves. I’ll derive that larger-data candidate in one closed-form solve and test it on hidden validation. The earlier validated edit remains backed up.

```python
# Hypothesis: a less-regularized direction estimated from all 900 practice items generalizes better than the 420-item correction.
delta900=np.linalg.solve(XX.T@XX+.1*np.eye(1024),XX.T@YY)
np.save(work/"scalar_delta_900_ridge01.npy",delta900)
print("new delta norm",np.linalg.norm(delta900),"old",np.linalg.norm(delta))
print("In-sample",np.mean(list(eval_all(delta900)[0].values())),eval_all(delta900)[1])
save_edit("scalar.weight",torch.from_numpy((W0+delta900).astype(np.float32)[None]))
with open(notes,"a") as f:
    f.write("\nOriginal reads now cover all 900. Three-fold practice CV (seed926): ridge .1=.69074, .3=.68519,1=.67259,3=.65926; no backbone training. Fitted new additive scalar direction on 900 with ridge .1."
            f"\nPass 9 validation: original+delta900_ridge.1, norm {np.linalg.norm(delta900):.4f}; all other tensors original. Earlier validated edit backed up.\n")
```
