The new solve produced NaNs because the read files use NaNs for padded option slots; multiplying those slots by zero did not remove them. No invalid weights were installed. I’ll replace only padding with zeros and rerun the same solve.
```python
# Hypothesis: NaNs are confined to non-path padding and can be removed without changing any observed head feature.
bad_slots = ~np.isfinite(X900).all(-1)
print("Nonfinite slots:", int(bad_slots.sum()), "in valid options:", int((bad_slots&mask900).sum()))
print("Valid nonfinite examples:", [(rows900[i]["type"], rows900[i]["options"][j]) for i,j in np.argwhere(bad_slots&mask900)[:5]])
X900 = np.where(mask900[...,None], X900, 0)
assert np.isfinite(X900).all() and np.isfinite(L900).all() and np.isfinite(G900).all()
Xc900 = center900(X900).astype(np.float32)
assert np.isfinite(Xc900).all()
d_mix900 = solve900(np.ones(len(rows900),bool), .006)
assert np.isfinite(d_mix900).all()
wmix900 = torch.from_numpy(w0+0.75*d_mix900).reshape(1,-1)
torch.save(torch.from_numpy(d_mix900), work/"candidate_weights"/"mix900_direction.pt")
torch.save(wmix900, work/"candidate_weights"/"mix900_075_scalar.pt")
full900_logits = L900+0.75*np.einsum("qod,d->qo",X900,d_mix900)
print("Full900 direction norm/cosine with600:", float(np.linalg.norm(d_mix900)),
      float(d_mix900@d_mix600/(np.linalg.norm(d_mix900)*np.linalg.norm(d_mix600))))
print("Full900 fitted (not held out):", evaluate900(full900_logits))
cross_logits = np.zeros_like(L900)
cross_dirs = []
for fold in range(3):
    held = folds900==fold
    d_cv = solve900(~held, .004)
    cross_dirs.append(d_cv)
    cross_logits[held] = L900[held]+0.75*np.einsum("qod,d->qo",X900[held],d_cv)
    print("Fixed-recipe held fold", fold, evaluate900(cross_logits, held))
cv900_summary = evaluate900(cross_logits)
print("Combined cross-fit:", cv900_summary)
note("\nCORRECTION: preceding full900/cross-fit numbers were invalid NaN artifacts from multiplying read-file NaN padding by zero. No invalid edit was installed. Padding now zeroed with np.where; all actual path features finite. Valid full900 in-sample "+str(evaluate900(full900_logits))+
     "; valid fixed-recipe cross-fit "+str(cv900_summary)+".")
import shutil, hashlib
if not (work/"best_2").exists():
    shutil.copytree(work/"edited",work/"best_2")
torch.save({"scalar.weight":load("scalar.weight").clone()},work/"best_2_weights.pt")
save_edit("scalar.weight",wmix900)
print("Installed scalar digest:", hashlib.sha256(wmix900.numpy().tobytes()).hexdigest())
note("\nPlanned forward17: paired hidden VALIDATION300. Exact edit scalar.weight=w_original+.75*d_mix900, candidate_weights/mix900_075_scalar.pt, lambda .006 fixed before viewing new features/labels; all other tensors original. No new recipe tuning. best2 backed up. All900 practice now used in solve; validation remains unseen.")
```
