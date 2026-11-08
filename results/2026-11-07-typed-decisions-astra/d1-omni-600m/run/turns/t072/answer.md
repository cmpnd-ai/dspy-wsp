The repair remains ahead under the metadata perturbation, but it is not fully invariant to irrelevant fields. I’ll record that limitation, verify that the installed tensors exactly match the validated repair, and run the final held-out validation check.
```python
# Hypothesis: the frozen, validation-confirmed repair remains the best supported choice despite measurable metadata sensitivity.
import shutil, hashlib
p24=torch.tensor(np.nan_to_num(np.load(W/"fwd_24"/"probabilities.npy")))
print("Actual edited robustness:",robustness_summary(p24))
print("Replay max probability error:",float((p24-p23_predicted).abs().max()))
robust_pairs={}
for variant in [False,True]:
    sel=(variant23==variant)
    bcorrect=(p23_original.argmax(-1)==gold23.argmax(-1)).float()
    ecorrect=(p24.argmax(-1)==gold23.argmax(-1)).float()
    diffs=[]
    for iid in robust_ids:
        js=[j for j,r in enumerate(m23["rows"]) if robust_tags[r["request"]]["id"]==iid and bool(variant23[j])==variant]
        diffs.append(float((ecorrect[js]-bcorrect[js]).mean()))
    aa=np.array(diffs)
    robust_pairs["metadata_added" if variant else "unchanged"]={
        "gain":float(aa.mean()),"paired_se":float(aa.std(ddof=1)/np.sqrt(len(aa)))}
print("Robustness paired gains:",robust_pairs)
final_names=["head.scorer.1.weight","head.scorer.1.bias","head.scorer.3.weight"]
assert set(edited())==set(final_names)
final_hashes={}
changed_entries=0
for name,tensor in zip(final_names,best2_params):
    current=load(name)
    assert torch.equal(current,tensor),name
    changed=int((current!=load_original(name)).sum())
    changed_entries+=changed
    final_hashes[name]={"sha256":hashlib.sha256(current.numpy().tobytes()).hexdigest(),
                        "changed_entries":changed,"shape":list(current.shape)}
print("Final modified entries:",changed_entries,"of 380732161")
shutil.copytree(W/"edited",W/"final_best",dirs_exist_ok=True)
manifest={
    "candidate":"conditional_candidate.pt",
    "fit_examples":448,"development_examples":224,"independent_practice_test_examples":228,
    "method":"Closed-form shared normalized-input residual plus 54 q/option-gated residual channels in 110 low-contribution scorer neurons.",
    "tensors":final_hashes,
    "development":{"original":.55,"edited":.7008928571428571,"gain":.15089285714285714,"paired_se":.01706094667315483},
    "independent_practice_test":{"original":.5842105263157894,"edited":.724561403508772,"gain":.14035087719298245,"paired_se":.016534041407624527},
    "validation_pass13":{"original":.5753333333333334,"edited":.7173333333333334,"gain":.142,"paired_se_rounded":.01543},
    "metadata_robustness":robust_pairs,
    "limitations":["Numerical invoice discrepancy magnitude remains weakly distinguished.",
                   "Irrelevant metadata can change answers; edited flip rate18.8% versus original13.6% on100 controlled pairs."]
}
json.dump(manifest,open(W/"final_manifest.json","w"),indent=2)
note("\nPass24 actual metadata perturbation replay: "+str(robustness_summary(p24))+"; paired gains "+str(robust_pairs)+". Repair remains ahead, but invariance is imperfect and worse than original by flip-rate metric. No tuning based on this diagnostic.\nFINAL AUDIT: exact best2 restored and checked tensor-equal to conditional_candidate.pt, only head.scorer.1.weight/.bias and head.scorer.3.weight edited;112860 entries total. Backup final_best/; hashes and measurements final_manifest.json.\nNext pass25 FINAL VALIDATION both300 hidden examples. Exact edit unchanged best2 (shared residual neurons33/746 plus54 gated residual pairs in108 low-contribution neurons); same as pass13. No further fitting or changes; expected deterministic repeat of57.533% ->71.733%, +14.200pp.")
```
