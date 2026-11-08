Nothing ran in that turn. I’ll retry the array conversion and coefficient check with a single code block; the installed weights remain unchanged.
```python
# Hypothesis: explicit array conversion will let us reproduce the original closed-form solution exactly.
def as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
print("Types:", {nm:(type(xx).__name__,str(xx.dtype)) for nm,xx in [("U",U),("fit_u",fit["u"]),("Y",Y),("Yt",Yt),("A_best",A_best)]})
print("U comparison:", float(np.max(np.abs(as_numpy(U)-as_numpy(fit["u"])))))
print("Target comparison:", float(np.max(np.abs(as_numpy(Y)-as_numpy(Yt)))))
print("Target function globals:", contrast_targets.__code__.co_names)
print("Combine function globals:", combine_data.__code__.co_names)
for nm in ["rank9_closed_form.npz","candidate_rank9_ridge1_factors.npz"]:
    with np.load(work_dir/nm) as zz:
        print(nm, [(k,zz[k].shape) for k in zz.files])
Y_dev, res_dev = contrast_targets(dev)
U_np = as_numpy(U).astype(np.float64)
Y_np = as_numpy(Y).astype(np.float64)
A_reproduced = np.linalg.solve(U_np.T @ U_np + np.eye(U_np.shape[1]), U_np.T @ Y_np).T
print("Reproduced coefficient relative error:", float(np.linalg.norm(A_reproduced-A_best)/np.linalg.norm(A_best)))
print("Dev target norm / linear residual median:", float(np.linalg.norm(Y_dev)), float(np.median(res_dev)))
```