The last local table included padded options, so its argmax values are invalid; the actual scored forward-pass results are unaffected. I’ll fix masking and reconstruct both saved edits from their tensors before testing workflow-specific directions.


```python
# Hypothesis: an exactly reconstructed, correctly masked scorer can test new readout directions without any backbone computation.
import torch.nn.functional as F
repair_mask=D["mask"].bool()
repair_features=torch.nan_to_num(D["features"].float())
repair_logits=torch.nan_to_num(D["option_logits"].float()).masked_fill(~repair_mask,-torch.inf)
repair_gold=D["gold"]
repair_target=D["target"]
repair_fit=D["isfit"].bool()
repair_tune=D["istune"].bool()
repair_workflows=sorted(set(q[0] for q in D["qkey"]))
repair_wfids=torch.tensor([repair_workflows.index(q[0]) for q in D["qkey"]])
repair_W0=load_original("residual_scorer.0.weight")
repair_b0=load_original("residual_scorer.0.bias")
repair_wout=load_original("residual_scorer.3.weight").flatten()
repair_gate=float(load_original("residual_gate").sigmoid())
repair_b1=torch.load(WORK/"linear_bypass_candidate.pt",weights_only=False)
repair_b2=torch.load(WORK/"targeted_candidate.pt",weights_only=False)
print("Saved keys:",repair_b1.keys(),repair_b2.keys())
def repair_simulate(tensors):
    rw,rb,ro=tensors
    ro=ro.flatten()
    changed=((rw!=repair_W0).any(1)|(rb!=repair_b0)|(ro!=repair_wout))
    ix=changed.nonzero().flatten()
    feature_batch=repair_features
    anew=F.gelu(F.linear(feature_batch,rw[ix],rb[ix]))
    aold=F.gelu(F.linear(feature_batch,repair_W0[ix],repair_b0[ix]))
    dlog=repair_gate*((anew*ro[ix]).sum(-1)-(aold*repair_wout[ix]).sum(-1))
    return (repair_logits+dlog).masked_fill(~repair_mask,-torch.inf)
def repair_measure(logits,subset,reference=repair_logits):
    p=logits.masked_fill(~repair_mask,-torch.inf).argmax(-1)
    b=reference.masked_fill(~repair_mask,-torch.inf).argmax(-1)
    ac=(p==repair_target).float()
    dif=ac-(b==repair_target).float()
    byitem=collections.defaultdict(list)
    for ii in subset.nonzero().flatten().tolist():
        byitem[D["ids"][ii]].append(float(dif[ii]))
    vals=np.array([np.mean(a) for a in byitem.values()])
    return dict(acc=float(ac[subset].mean()),gain=float(vals.mean()),se=float(vals.std(ddof=1)/np.sqrt(len(vals))),n=len(vals))
print("Stored tensor shapes:", {str(k):tuple(v.shape) if isinstance(v,torch.Tensor) else type(v).__name__ for k,v in repair_b1.items()})
repair_current_tensors=tuple(load(n) for n in ["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"])
repair_current_logits=repair_simulate(repair_current_tensors)
print("Current tune reconstruction:",repair_measure(repair_current_logits,repair_tune))
print("Fit/tune padded mask agreement:",torch.equal(repair_mask,D["option_mask"]))
print("Corrected tune by field, original/current:")
for qk in qkeys:
    sub=torch.tensor([z==qk for z in D["qkey"]])&repair_tune
    print(qk,[round(repair_measure(z,sub)["acc"],3) for z in [repair_logits,repair_current_logits]])
```
