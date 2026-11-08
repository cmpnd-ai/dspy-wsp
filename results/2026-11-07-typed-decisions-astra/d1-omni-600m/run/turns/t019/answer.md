I’ll compare the candidate’s predicted cue responses with the original, then install it at quarter strength. The edit will replace only two nearly unused scorer neurons; a positive/negative GELU pair carries the residual direction exactly, leaving all other neurons and the backbone unchanged.
```python
# Hypothesis: routing the missing residual through scorer neurons 33 and 746 causally restores cue sensitivity, with an observable dose response.
cfmeta=json.load(open(WORK/"fwd_6/meta.json"))["rows"]
cfa={p.stem:np.load(p) for p in (WORK/"fwd_6").glob("*.npy")}
cfz,cfx,_=scorer_features(cfa["head_inputs"])
cfz=torch.nan_to_num(cfz); cfx=torch.nan_to_num(cfx)
cfmask=torch.tensor(cfa["option_mask"])
cfbase=torch.tensor(np.nan_to_num(cfa["option_logits"],nan=0))
cflogits=cfbase-cfz[:,:,pair_units]@W0.squeeze()[pair_units]+cfx@candidate_dx
def cf_values(logits):
    vals={}
    for j,r in enumerate(cfmeta):
        n=len(r["options"])
        t=cfg["temperatures"][r["type"]+":2" if r["type"]=="noul" else r["type"]+":3-5"]
        p=torch.softmax(logits[j,:n]/t,0)
        if r["type"]=="score":value=float(p@torch.arange(n,dtype=p.dtype))
        elif r["type"]=="noul":value=float(p[1])
        elif r["question"]=="outcome":value=float(p[r["options"].index("harmful")])
        elif r["question"]=="action" and "stop" in r["options"]:value=float(p[r["options"].index("stop")])
        else:continue
        tag=cf_tags[r["request"]]
        vals[(tag["family"],tag["id"],str(tag["value"]),r["question"])]=value
    return vals
cfvb=cf_values(cfbase); cfve=cf_values(cflogits)
cfcontrasts=[
    ("constraint_violations","0","3","outcome"),
    ("constraint_violations","0","3","risk"),
    ("constraint_violations","0","3","needs_review"),
    ("days_until_due","30","-5","urgency"),
    ("discount","None","2","urgency"),
    ("freight_terms","billable","prepaid","discrepancy_severity"),
    ("privileges","read","production","severity"),
    ("asset_criticality","low","critical","severity")]
for family,low,high,q in cfcontrasts:
    ids=sorted({tag["id"] for tag in cf_tags if tag["family"]==family})
    b=np.array([cfvb[(family,i,high,q)]-cfvb[(family,i,low,q)] for i in ids])
    e=np.array([cfve[(family,i,high,q)]-cfve[(family,i,low,q)] for i in ids])
    print(family,q,"original",round(b.mean(),4),"proposed",round(e.mean(),4),"positive pairs",sum(b>0),"/",sum(e>0))
def install_channel(direction, strength, tag):
    w1=scorer_params["1.weight"].clone()
    b1=scorer_params["1.bias"].clone()
    w3=scorer_params["3.weight"].clone()
    scale=float(direction.norm())
    a=direction/scale
    bias=-float(a@mu_x)
    j,k=pair_units
    w1[j]=a; w1[k]=-a
    b1[j]=bias; b1[k]=-bias
    w3[0,j]=strength*scale; w3[0,k]=-strength*scale
    save_edit("head.scorer.1.weight",w1)
    save_edit("head.scorer.1.bias",b1)
    save_edit("head.scorer.3.weight",w3)
    torch.save({"direction":direction,"strength":strength,"units":pair_units,"mu_x":mu_x},WORK/(tag+"_definition.pt"))
    return w1,b1,w3
w1edit,b1edit,w3edit=install_channel(candidate_dx,0.25,"candidate_quarter")
probe=F.linear(F.gelu(F.linear(D["x"],w1edit,b1edit)),w3edit,scorer_params["3.bias"]).squeeze(-1)
predicted=D["logits"]-pair_removed+0.25*(D["x"]@candidate_dx-candidate_dx@mu_x)
print("Edited scorer reconstruction error:",float((centered(probe)-centered(predicted)).abs().max()))
json.dump({"type":"decide","model":"both","problems":splits["dev"]},open(WORK/"dose_dev.json","w"))
note("Counterfactual source pass 6 read; predicted edit cue effects printed and saved in kernel. Current exact edits: head.scorer.1.weight rows 33/746 = +/- candidate_dx/||candidate_dx||; biases = -/+ projection of fit input mean; head.scorer.3.weight columns 33/746 = +/- 0.25*||candidate_dx||. All other values original. Definition candidate_quarter_definition.pt.\n"
     "Next pass: DECIDE both on 224 dev IDs in dose_dev.json, strength .25. Preregistered local prediction: accuracy 59.821%, +4.821pp, paired SE 1.055pp.")
```
