The context-conditioned key repair gives only a small development gain over the validated rank-one edit. I’ll test this fixed candidate on the reserved practice cases before deciding whether the extra directions are worthwhile.

```python
# Hypothesis: conditioning option evidence on four existing query modes may improve independent cases, but must justify its extra complexity.
jl,js=best_joint[:2]
jc=joint_candidates[jl]*js
deltaK_joint=u[:,None]@jc[:2048][None,:]+Uctx@jc[2048:].reshape(rctx,popt)@Popt.T
Ljoint_dev=dev["L"]+(torch.einsum("nkd,nd->nk",
    torch.nn.functional.linear(dev["o"],deltaK_joint),dev["Q"])/math.sqrt(768))
pred_joint=dev["L"]+js*(Jdev@joint_candidates[jl])*dev["temp"][:,None]
print("Joint tensor exact centered-logit reproduction error:",
      center_options(Ljoint_dev-pred_joint,dev["mask"])[dev["mask"]].abs().max().item())
print("Joint delta norm",deltaK_joint.norm().item(),"rank at most",1+rctx)
jointdir=WORK/"candidate_joint_key"
jointdir.mkdir(exist_ok=True)
torch.save({"deltaK":deltaK_joint,"lambda":jl,"strength":js,"u":u,"Uctx":Uctx,"Popt":Popt,
            "coefficients":jc,"fit_ids":splits["fit"]},jointdir/"derivation.pt")
save_edit("k.weight",H0["k.weight"]+deltaK_joint)
lognote("## Before pass 15\nExact edit: k.weight = K0 + deltaK_joint; rank <=5, deltaK_joint in "
        "candidate_joint_key/derivation.pt. Joint closed-form lambda .001, strength1, fit540 only, "
        "floor .1. All other tensors original. Paired DECIDE reserved180 test IDs in paired_test.json. "
        "Development gain over best2 only .006667 SE .010559; test will determine whether extra context paths merit validation.")
```
