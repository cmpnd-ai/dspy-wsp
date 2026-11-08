I’ll first verify that the closed-form calculation reproduces the saved coefficients. Then I’ll use that same calculation once on the expanded sample. This avoids introducing a new tuning loop.
```python
# Hypothesis: the same fixed ridge solution benefits from 540 rather than 360 labeled records.
print("U comparison:", np.max(np.abs(U-fit["u"])), U.dtype, Y.dtype, Yt.dtype)
print("Target comparisons:", float(np.max(np.abs(Y-Yt))))
print("Key norms:", {nm:round(float(np.linalg.norm(xx)),5) for nm,xx in [("U",U),("Y",Y),("Yt",Yt),("A_best",A_best)]})
print("Target function globals:", contrast_targets.__code__.co_names)
print("Combine function globals:", combine_data.__code__.co_names)
print("Factors saved:", [(nm, [(k, z[k].shape) for k in z.files]) for nm in ["rank9_closed_form.npz","candidate_rank9_ridge1_factors.npz"] for z in [np.load(work_dir/nm)]])
Y_dev, res_dev = contrast_targets(dev)
A_reproduced = np.linalg.solve(U.T @ U + np.eye(U.shape[1]), U.T @ Y).T
print("Reproduced coefficient relative error:", float(np.linalg.norm(A_reproduced-A_best)/np.linalg.norm(A_best)))
print("Dev target norm / linear residual median:", float(np.linalg.norm(Y_dev)), float(np.median(res_dev)))
```