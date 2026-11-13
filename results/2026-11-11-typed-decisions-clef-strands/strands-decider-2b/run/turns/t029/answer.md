I’ll quantify the cue-induced margin changes and verify the edited model on the same probes. In parallel, I’ll inspect whether the remaining errors could benefit from a small query-side correction that uses information available only at the final decision token.

```python
# Hypothesis: the validated key edit improves cue sensitivity, while residual errors may reflect underused decision-token information.
def load_unlabelled_read(passnum):
    root=WORK/f"fwd_{passnum}"
    meta=json.load(open(root/"meta.json"))["rows"]
    mask=torch.from_numpy(np.load(root/"option_mask.npy"))
    d=ln0(torch.from_numpy(np.load(root/"decide.npy")[:,0,:].copy()))
    o=ln0(torch.from_numpy(np.load(root/"head_inputs.npy")).nan_to_num())
    L=torch.from_numpy(np.load(root/"option_logits.npy")).nan_to_num()
    return {"d":d,"o":o,"mask":mask,"L":L,"meta":meta,
            "temp":torch.tensor([temps[m["type"]] for m in meta])}
cf=load_unlabelled_read(11)
cf["Q"]=head_q(cf["d"])
cf["g"]=cf["Q"]@u/math.sqrt(768)
cfC=correction(cf,v0)
cf_lookup={(m["request"],m["question"]):i for i,m in enumerate(cf["meta"])}
cue_stats=collections.defaultdict(list)
for pair in counter_meta:
    a,b=[cf_lookup[(r,pair["qid"])] for r in pair["requests"]]
    opts=cf["meta"][a]["options"]
    ip,im=opts.index(pair["positive"]),opts.index(pair["negative"])
    old=((cf["L"][b,ip]-cf["L"][b,im])-(cf["L"][a,ip]-cf["L"][a,im])).item()/cf["temp"][a].item()
    new=((cf["L"][b,ip]+cfC[b,ip]-cf["L"][b,im]-cfC[b,im])-
         (cf["L"][a,ip]+cfC[a,ip]-cf["L"][a,im]-cfC[a,im])).item()/cf["temp"][a].item()
    cue_stats[pair["cue"]].append((old,new))
for cue,ss in cue_stats.items():
    a=np.array(ss)
    print(cue,"mean expected-sign log-odds movement original/edited",np.round(a.mean(0),3),
          "positive pairs", (a>0).sum(0).tolist(),"/",len(a))
# Inspect input/output spaces for a bounded query-side correction orthogonal to the existing key edit.
dmean=fit["d"].mean(0)
Dc=fit["d"]-dmean
deig,dvec=torch.linalg.eigh(Dc.T.double()@Dc.double()/len(Dc))
print("Decision residual variance captured by PCs:",
      {r:round((deig[-r:].sum()/deig.sum()).item(),4) for r in [8,16,32,64,128]})
Kfit=head_k(fit["o"])
Kc=center_options(Kfit,fit["mask"])
Korth=Kc-(Kc@u)[:,:,None]*u
Kg=Korth[fit["mask"]]
keig,kvec=torch.linalg.eigh(Kg.T.double()@Kg.double()/len(Kg))
print("Orthogonal option-key variance captured by PCs:",
      {r:round((keig[-r:].sum()/keig.sum()).item(),4) for r in [4,8,16,32,64]})
lognote("Counterfactual predicted sensitivity (positive means expected direction):\n"+"\n".join(
    f"- {cue}: mean original/edited={np.array(ss).mean(0).tolist()}, "
    f"positive pairs={(np.array(ss)>0).sum(0).tolist()}/10" for cue,ss in cue_stats.items()))
lognote("## Before pass 12\nExact edit unchanged: validated K0 + outer(u,v), lambda .001; other tensors original. "
        "READ edited on same 80 single-cue counterfactual requests in counterfactual_edited.json. "
        "Expected response predicted from pass 11 cached head inputs, now verify actual edited output.")
```
