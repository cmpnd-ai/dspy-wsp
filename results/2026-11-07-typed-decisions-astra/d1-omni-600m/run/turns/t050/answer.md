The cue-restricted fits did not improve development accuracy meaningfully—at most 0.09 percentage points—so I won’t add them. I’ll instead test the validated conditional repair at half and double strength to check its dose response and guard against overcorrection.
First I’ll calculate the expected dose response from the saved scorer inputs, then install only the half-strength conditional extension. The shared residual will stay at its validated strength.
```python
# Hypothesis: the conditional correction has a finite useful dose; too little misses cues and too much overcorrects.
def paired_local(a,b,split="dev"):
    good_a=(a.masked_fill(~D["mask"],-1e6).argmax(-1)==D["gold"].argmax(-1)).float()
    good_b=(b.masked_fill(~D["mask"],-1e6).argmax(-1)==D["gold"].argmax(-1)).float()
    positions={}
    for j,r in enumerate(D["rows"]):
        if D[split][j]:
            positions.setdefault(r["id"],[]).append(j)
    dif=np.array([float((good_a[ix]-good_b[ix]).mean()) for ix in positions.values()])
    return {"gain":float(dif.mean()),"se":float(dif.std(ddof=1)/np.sqrt(len(dif))),
            "improved":int((dif>0).sum()),"worsened":int((dif<0).sum()),"n":len(dif)}

best2_params=tuple(t.clone() for t in cc["params"])
best2_logits=cond_logits.clone()
half_cond_params=make_conditional(strength=.5)
print("Half changes input rows?",float((half_cond_params[0]-best2_params[0]).abs().max()))
half_cond_logits=logits_scorer_params(D["x"],half_cond_params)
double_cond_params=make_conditional(strength=2.)
double_cond_logits=logits_scorer_params(D["x"],double_cond_params)
for strength,ll in [(.5,half_cond_logits),(1.,best2_logits),(2.,double_cond_logits)]:
    print("Conditional strength",strength,evaluate_logits(ll),"vsbest2",paired_local(ll,best2_logits))
torch.save({"params":half_cond_params,"strength":.5,"base":"conditional_candidate.pt"},W/"conditional_half.pt")
torch.save({"params":double_cond_params,"strength":2.,"base":"conditional_candidate.pt"},W/"conditional_double.pt")
save_scorer_params(half_cond_params)
note("\nInvoice cue subspace experiment: paired numeric/text cue directions fit with closed-form residual ridge on448 fit examples only; max dev increment0.089pp, not sufficient. Rejected; no additional cue-fit weights saved.\nNext pass16: DECIDE both224 dev IDs in dose_dev.json. Exact edit conditional_half.pt: base linear pair33/746 unchanged at1x; same54 q/option gated directions on108 low-contribution neurons, their final output entries halved to +/-10 (conditional extension strength.5); other parameters unchanged from conditional_candidate.pt. Predicted "+str(evaluate_logits(half_cond_logits))+". best_2 remains backed up.")
```
