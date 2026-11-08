The full-900 refit reached 57.93% on hidden validation, a gain of 23.13 ± 1.47 percentage points over the original. I’ll preserve it, then check the correction’s strength response and remaining errors using cached cross-fit predictions only.
```python
# Hypothesis: the repaired readout has a stable strength range, while remaining failures concentrate in particular question/cue families.
note("\nForward17 hidden validation300 full900 mix(.75): original .348000, edited .579333, paired gain .231333 SE .014716;205 improved/25 worsened. New best confirmed candidate; +.005333 over best2 not claimed significant.")
if not (work/"best_3").exists():
    shutil.copytree(work/"edited",work/"best_3")
torch.save({"scalar.weight":load("scalar.weight").clone()},work/"best_3_weights.pt")
wf900 = np.array([r["workflow"] for r in rows900])
qid900 = np.array([r["question"] for r in rows900])
cross_correction = (cross_logits-L900)/0.75
strength_results = []
for strength in [0, .25, .5, .625, .75, .875, 1, 1.25]:
    ll = L900+strength*cross_correction
    result = evaluate900(ll)
    strength_results.append({"strength":strength,**result})
    print("Cross-fit strength",strength, {k:round(v,5) if isinstance(v,float) else v for k,v in result.items()})
print("Per-question cross-fit original -> repaired:")
for wf in sorted(set(wf900)):
    for qid in sorted(set(qid900[wf900==wf])):
        sel = (wf900==wf)&(qid900==qid)
        orig = evaluate900(L900,sel)
        rep = evaluate900(cross_logits,sel)
        print(wf,qid, round(orig["accuracy"],3),"->",round(rep["accuracy"],3),
              "gain",round(rep["gain"],3))
correct_cross = np.where(mask900,cross_logits,-1e20).argmax(1)==G900.argmax(1)
gold_sorted = np.sort(G900,axis=1)
gold_margin = gold_sorted[:,-1]-gold_sorted[:,-2]
print("Accuracy by gold top-two gap:", [(a,b,round(float(correct_cross[(gold_margin>=a)&(gold_margin<b)].mean()),3),
                                        int(((gold_margin>=a)&(gold_margin<b)).sum()))
                                       for a,b in [(0,.1),(.1,.3),(.3,.6),(.6,1.01)]])
print("Existing target families:", list(target_specs), list(tangent_specs))
note("\nFixed-recipe cross-fit strength diagnostic (head only; no forward and no weight fitting): "+str(strength_results)+
     ". best3 full900 strength.75 backed up.")
```
