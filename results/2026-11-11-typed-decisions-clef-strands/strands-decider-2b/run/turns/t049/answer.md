The actual low-energy-only edit retained almost the entire gain, supporting the readout diagnosis. I’ll restore the validated repair and test one final closed-form alternative: smooth the gold probabilities rather than flooring them. This may preserve distinctions between weak options without demanding extreme log odds. All candidate directions will use fitting data only.


```python
# Hypothesis: smoothly bounded probability targets may improve the same state-evidence readout without adding parameters or context paths.
save_edit("k.weight", torch.from_numpy(np.load(WORK/"best_2/edited/k.weight.npy")).float())
# Recover the exact sufficient-statistic convention before reusing the existing fit-only eigensystem.
Xraw_check = Xfit3[fit["mask"]]
print("Existing X convention:", float((X-Xraw_check).abs().max()),
      "weighted:",float((X-Xraw_check*weights[:,None]).abs().max()))
print("weights range",float(weights.min()),float(weights.max()))
for j in [0,10,500]:
    print("Gram diagonal",j,float(gram[j,j]),"X squared sum",float(X[:,j].double().square().sum()))
y_ref = center_options(fit["gold"].clamp_min(.1).log()-fit["L"]/fit["temp"][:,None],fit["mask"])[fit["mask"]]
rhs_ref_check = X.double().T @ (y_ref*weights).double()
v_ref_check = (evecs @ ((evecs.T @ rhs_ref_check)/(evals+.0003))).float() * .75
print("Reconstructed v_ref error:",float((v_ref_check-v_ref).norm()/v_ref.norm()))
print("Ridge system scale:",float(evals.max()),float(evals.min()),"number rows",len(X))
with open(WORK/"notes.md","a") as f:
    f.write("\nPass 17 actual low-energy-only ablation: development .577778 -> .771111, "
            "gain .193333, SE .018097; 117 improved, 22 worsened. Full best2 is .772222 cached. "
            "Thus low-energy option-state readout accounts for essentially all benefit. Restored exact best2.\n")
    f.write("Final planned fit-only closed-form target comparison, same rank1 mechanism and no new subspaces: "
            "log(p+.05), log(p+.1), log(p+.2), and centered 4*p. "
            "Ridges .0003/.001 and strengths .75/1 fixed in advance; select on dev only, "
            "advance only if improvement over best2 is larger than paired dev uncertainty.\n")
```
