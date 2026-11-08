The calculation reproduces the saved coefficients to numerical precision. I’ll now solve the same ridge system on 540 records, install the resulting rank-nine update, and test it on the 360 records still excluded from fitting.
```python
# Hypothesis: expanding the fixed closed-form fit reduces state-dependent coefficient variance and improves held-out accuracy.
U_expanded = np.concatenate([as_numpy(fit["u"]), as_numpy(dev["u"])], axis=0).astype(np.float64)
Y_expanded = np.concatenate([Y_np, as_numpy(Y_dev)], axis=0).astype(np.float64)
A_expanded = np.linalg.solve(
    U_expanded.T @ U_expanded + np.eye(U_expanded.shape[1]),
    U_expanded.T @ Y_expanded
).T
delta_expanded = torch.from_numpy(B.astype(np.float32)) @ (
    torch.from_numpy(A_expanded.astype(np.float32)) @ w2orig
)
w2_name = "model.language_model.layers.23.feed_forward.w2.weight"
save_edit(w2_name, w2orig + delta_expanded)
np.savez(work_dir/"candidate_expanded_factors.npz", B=B, A=A_expanded, ridge=1.0, alpha=1.0,
         fit_ids=np.array(split["fit"]+split["dev"]))
if (work_dir/"candidate_expanded").exists():
    shutil.rmtree(work_dir/"candidate_expanded")
shutil.copytree(work_dir/"edited", work_dir/"candidate_expanded")
print("Expanded coefficient norm:", float(np.linalg.norm(A_expanded)))
print("Relative update norm:", float(torch.linalg.vector_norm(delta_expanded)/torch.linalg.vector_norm(w2orig)))
print("Expanded in-fit dev metrics (NOT holdout):", metrics_array(dev, recompute_head(dev,A_expanded)[1]))
with (work_dir/"notes.md").open("a") as f:
    f.write("\nOriginal ridge solution reproduced at relative coefficient error 2.85e-10. Expanded fit solved once, ridge1 alpha1, 540 records (fit+dev), 2700 questions; no test coefficients or validation labels used. Saved candidate_expanded/ and candidate_expanded_factors.npz.\n")
    f.write("\nPlanned pass 19: DECIDE both on the 360 reserved practice-test IDs (test_pair.json). Exact edit: W2_original+B A_expanded540 W2_original, rank<=9; all other tensors original. Coefficients use 540 fit+dev records, fixed original target, ridge1, alpha1.\n")
```