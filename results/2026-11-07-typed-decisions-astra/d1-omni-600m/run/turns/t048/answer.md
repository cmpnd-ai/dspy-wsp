The fitted correction is mostly state-dependent, not a fixed preference for particular options. I’ll now quantify which invoice cues it actually responds to and whether numerical mismatch evidence is present at the scorer input.
```python
# Hypothesis: invoice condition words carry a usable direction, while numeric equality may remain a separate unresolved mechanism.
import torch.nn.functional as F
input15=torch.tensor(np.nan_to_num(in15),dtype=torch.float32)
x15=F.layer_norm(input15,(1024,),scorer_params["0.weight"],scorer_params["0.bias"])
mask15=torch.tensor(np.load(W/"fwd_15"/"option_mask.npy"))
base15=torch.tensor(np.nan_to_num(np.load(W/"fwd_15"/"option_logits.npy")))
edited15=logits_scorer_params(x15,cond_params)
shared_params=torch.load(W/"candidate_full_definition.pt",weights_only=False)
print("Shared definition keys",shared_params.keys())
temp15=torch.tensor([cfg["temperatures"]["noul:2" if r["type"]=="noul" else (r["type"]+":3-5")] for r in m15["rows"]])
def summarize_invoice_cues(logits):
    pp=torch.softmax((logits/temp15[:,None]).masked_fill(~mask15,-1e6),-1)
    out={}
    for j,row in enumerate(m15["rows"]):
        tag=inv_probe_tags[row["request"]]
        if row["question"]=="discrepancy_severity":
            val=float(pp[j,:4]@torch.arange(4,dtype=torch.float32))
        elif row["question"]=="matches_order":
            val=float(pp[j,1])
        elif row["question"]=="disposition":
            val=float(pp[j,0])
        else:
            continue
        out.setdefault((tag["family"],str(tag["value"]),row["question"]),[]).append(val)
    return {k:float(np.mean(v)) for k,v in out.items()}
cv15b=summarize_invoice_cues(base15)
cv15e=summarize_invoice_cues(edited15)
print("Mean discrepancy level, matches_order P(true), approve probability: original -> best2")
for family,value in [("matched",0),("price",1.01),("price",1.1),("price",1.4),("billed_qty",1.1),("billed_qty",1.4),
                     ("received_qty",0.9),("received_qty",0.6),("condition","accepted with exceptions"),("condition","damaged"),("condition","rejected")]:
    print(family,value,[(q,round(cv15b[(family,str(value),q)],3),round(cv15e[(family,str(value),q)],3)) for q in ["discrepancy_severity","matches_order","disposition"]])
print("Input distances from matched by cue (normalized scorer inputs, discrepancy question)")
rr=[i for i,row in enumerate(m15["rows"]) if row["question"]=="discrepancy_severity"]
for t in range(1,11):
    distances=[]
    for ii in range(8):
        distances.append(float((x15[rr[ii*11+t]]-x15[rr[ii*11]]).square().mean().sqrt()))
    print(inv_probe_tags[t]["family"],inv_probe_tags[t]["value"],round(np.mean(distances),4))
note("\nPass15: controlled invoice cue baseline obtained (8 fit sources x11 variants,440questions). Current best2 unchanged. Testing numerical vs textual cue sensitivity using original scorer inputs; no backbone simulation or new fit labels.")
```
