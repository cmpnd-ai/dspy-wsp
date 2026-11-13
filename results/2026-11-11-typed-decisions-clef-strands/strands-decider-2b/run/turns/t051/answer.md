The linear-probability target has a small development advantage, but it is not decisive. I’ll fix that candidate now and test it on the reserved practice cases. The validation-confirmed edit remains the fallback; I won’t use individual test errors to revise the candidate.


```python
# Hypothesis: linear probability contrasts preserve useful option rankings with less distortion from small gold probabilities.
v_linear = smooth_candidates[("4*p",.001,.75)].clone()
Cdev_linear = correction(dev,v_linear)
Cfit_linear = correction(fit,v_linear)
print("Candidate fit:",eval_cached(fit,fit["L"]+Cfit_linear))
print("Candidate development:",eval_cached(dev,dev["L"]+Cdev_linear))
print("Direction cosine with best2:",torch.nn.functional.cosine_similarity(v_linear,v_ref,dim=0).item())
print("Norm ratio:",float(v_linear.norm()/v_ref.norm()))
print("Development dose response (same closed-form direction):")
for dose in (0.,.5,1.,1.5):
    print(dose,eval_cached(dev,dev["L"]+dose*Cdev_linear))
(WORK/"candidate_linear_rank1").mkdir(exist_ok=True)
torch.save({"u":u,"v":v_linear,"target":"4*p","ridge":.001,"strength":.75,
            "fit_ids":splits["fit"],"dev_ids":splits["dev"]},
           WORK/"candidate_linear_rank1/derivation.pt")
save_edit("k.weight",K0+torch.outer(u,v_linear))
with open(WORK/"notes.md","a") as f:
    f.write("\nSelected final target candidate using development only: target 4*p, ridge .001, strength .75. "
            "Small dev increment .010000 SE .007268 over best2; not asserted significant. "
            "Fit-only derivation remains rank1, independent solve, no training. "
            f"Cosine with best2 {torch.nn.functional.cosine_similarity(v_linear,v_ref,dim=0).item():.6f}.\n")
    f.write("\n## Before pass 18\nExact edit: k.weight = K0 + outer(u,v_linear), "
            "v_linear is .75 times ridge(lambda=.001) solve for centered [4*p - original logits/temperature], "
            "all other tensors original. Derivation in candidate_linear_rank1/derivation.pt. "
            "Paired DECIDE reserved 180 practice test cases in paired_test.json. "
            "No individual test errors used; best2 remains saved fallback.\n")
```
