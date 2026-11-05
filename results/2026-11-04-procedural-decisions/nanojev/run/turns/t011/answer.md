With labels aligned, the correction improves internal held-out accuracy from 49.6% to 61.9%, especially severity and workflow impact. I’ll use ridge strength 1, refit once on the 420 fitting examples, and test half-strength first. The edit changes only the scalar evidence direction, leaving the backbone and set-attention weights intact.

```python
# Hypothesis: a half-strength evidence-direction edit causally reduces answer-prior errors on unseen practice records.
lam=1.
delta=np.linalg.solve(X.T@X+lam*np.eye(1024),X.T@y)
print("Delta norm",np.linalg.norm(delta),"original norm",np.linalg.norm(W0),"cosine",
      np.dot(delta,W0)/(np.linalg.norm(delta)*np.linalg.norm(W0)))
for strength in [0,.25,.5,1,1.5,2]:
    print("strength",strength,"fit score",scores_direction(delta,strength))
save_edit("scalar.weight",torch.from_numpy((W0+.5*delta).astype(np.float32)[None]))
test450=test_ids[:150]+test_ids[160:310]+test_ids[320:470]
assert len(test450)==450 and not (set(test450)&set(fit_ids))
np.save(work/"scalar_delta_ridge1.npy",delta)
with open(notes,"a") as f:
    f.write("\nRead order checked: baseline order, not requested ID order. Initial misaligned exploratory calculations discarded, no weights applied."
            "\nClosed-form residual fit: within-question centered multiclass contrasts; boolean true vs fixed zero; ridge=1 selected on internal 84-item holdout (.6190 vs .4960 baseline). Refit on 420 fitting examples."
            f"\nPass 3: scalar.weight = original + 0.5 * delta (delta norm {np.linalg.norm(delta):.4f}); all other tensors original. Paired test on 450 disjoint practice items.\n")
json.dump({"type":"decide","model":"both","problems":test450},open(work/"test_request.json","w"))
```
