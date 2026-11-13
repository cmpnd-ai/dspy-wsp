The query-side addition is not convincingly better, so I’m leaving it out. A revised rank-one solve that gives less weight to near-zero gold probabilities improves development accuracy to 77.22%. I’ll test that fixed candidate independently, keeping the validated edit safely backed up.

```python
# Hypothesis: emphasizing distinctions among plausible options reduces rank-one readout overcorrection and improves top-choice accuracy.
Cfit_ref=correction(fit,v_ref)
Cdev_ref=correction(dev,v_ref)
ref_dir=WORK/"candidate_rank1_refined"
ref_dir.mkdir(exist_ok=True)
torch.save({"u":u,"v":v_ref,"floor":floor_ref,"lambda":lam_ref,"strength":s_ref,
            "fit_ids":splits["fit"]},ref_dir/"derivation.pt")
save_edit("k.weight",H0["k.weight"]+u[:,None]@v_ref[None,:])
print("Refined correction vs validated direction cosine:",
      torch.nn.functional.cosine_similarity(v_ref,v0,dim=0).item())
print("Refined dev vs original:",accuracy_stats(dev,dev["L"]+Cdev_ref))
print("Refined dev vs validated:",paired_between(dev,dev["L"]+Cdev,dev["L"]+Cdev_ref))
lognote("## Before pass 13\nExact edit: k.weight = K0 + outer(u,v_ref); v_ref is .75 times fit-only ridge "
        "solution lambda .0003 with log-gold floor .1, saved in candidate_rank1_refined/derivation.pt. "
        "All other tensors original. Refined candidate fixed using fit/dev only. "
        "Paired DECIDE on reserved 180 practice-test cases in paired_test.json. "
        "Previous test was used only to confirm best_1; its individual errors were not inspected or used in refinement.")
```
