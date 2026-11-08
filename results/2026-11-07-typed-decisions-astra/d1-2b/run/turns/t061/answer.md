I’ll remove the diagnostic norm edit, recover the remaining write coordinates, and solve the same fixed ridge system on 720 practice records. Then I’ll install the rank-nine repair and evaluate the 180 records excluded from this fit.
```python
# Hypothesis: a larger fixed-recipe closed-form fit improves the same state-dependent option-contrast repair.
extra_odd = read_arrays(23)
extra_co, extra_scale_o, extra_anchor_o = infer_subset(extra_orig, extra_odd, odd_idx)
extra_c = np.zeros_like(extra_orig["h"], dtype=np.float32)
extra_c[:, even_idx] = extra_ce
extra_c[:, odd_idx] = extra_co
extra_orig["c"] = extra_c
extra_orig["u"] = extra_c / gp_np[None, :]
extra_orig["temp"] = temp_for_data(extra_orig)
Y_extra, residual_extra = contrast_targets(extra_orig)
U_final = np.concatenate([U_expanded, extra_orig["u"]], axis=0).astype(np.float64)
Y_final = np.concatenate([Y_expanded, Y_extra], axis=0).astype(np.float64)
A_final720 = np.linalg.solve(
    U_final.T @ U_final + np.eye(U_final.shape[1]),
    U_final.T @ Y_final
).T
delta_final720 = torch.from_numpy(B.astype(np.float32)) @ (
    torch.from_numpy(A_final720.astype(np.float32)) @ w2orig
)
drop_edit(post_name)
save_edit(w2_name, w2orig + delta_final720)
np.savez(work_dir/"candidate_final720_factors.npz",
         B=B, A=A_final720, ridge=1.0, alpha=1.0,
         fit_ids=np.array(split["fit"] + split["dev"] + expand_ids))
if (work_dir/"candidate_final720").exists():
    shutil.rmtree(work_dir/"candidate_final720")
shutil.copytree(work_dir/"edited", work_dir/"candidate_final720")
json.dump({"type":"decide","model":"both","problems":final_check_ids},
          (work_dir/"final_check_pair.json").open("w"))
print("Odd probe anchor RMS error:", float(extra_anchor_o))
print("Fit dimensions:", U_final.shape, Y_final.shape)
print("Coefficient rank:", np.linalg.matrix_rank(A_final720))
print("Relative W2 update norm:", float(torch.linalg.vector_norm(delta_final720)/torch.linalg.vector_norm(w2orig)))
print("Current edits:", edited())
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 23 odd-coordinate probe completed, anchor RMS error %.3g. Diagnostic norm edit removed. Recovered write features for 180 extra practice records. Final closed-form fit uses 720 records / 3600 questions; original basis, target, ridge1, alpha1 unchanged. Candidate saved in candidate_final720/ with factors.\n" % float(extra_anchor_o))
    f.write("\nPlanned pass 24: DECIDE both on remaining 180 coefficient-excluded practice records in final_check_pair.json. Exact edit: layers.23.feed_forward.w2.weight = W2_original + B A_final720 W2_original, rank<=9, ridge1, alpha1. All other tensors original. These records were previously evaluated for other candidates but never used for coefficient estimation; sample-size extension was predeclared.\n")
```