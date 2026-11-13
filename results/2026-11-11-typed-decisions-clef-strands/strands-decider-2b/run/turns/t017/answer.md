The corrected rank-one edit raises development accuracy from 57.8% to 74.8%. I’ll now test its mechanism: compare state-dependent corrections with a static option-bias control, then apply half strength for an actual paired forward test.

```python
# Hypothesis: the new key direction uses state-dependent evidence, not just fixed preferences for option names.
def correction(data,v):
    return data["g"][:,None]*(data["o"]@v)
Cfit=correction(fit,v0)
Cdev=correction(dev,v0)
schema_means={}
for g in group_order:
    ix=torch.tensor([i for i,x in enumerate(fit["group"]) if x==g])
    schema_means[g]=center_options(Cfit[ix],fit["mask"][ix]).mean(0)
Cdev_static=torch.stack([schema_means[g] for g in dev["group"]])
print("Development controls:")
for name,LL in [("original",dev["L"]),("static fit option offsets",dev["L"]+Cdev_static),
                 ("full state-dependent direction",dev["L"]+Cdev)]:
    print(name,accuracy_stats(dev,LL))
rng_ctrl=np.random.default_rng(915)
Cshuf=Cdev.clone()
for g in group_order:
    ix=np.array([i for i,x in enumerate(dev["group"]) if x==g])
    Cshuf[ix]=Cdev[rng_ctrl.permutation(ix)]
print("Within-question shuffled state corrections:",accuracy_stats(dev,dev["L"]+Cshuf))
energy_by_dim=(Xfit3*v0[None,None,:]).square().sum((0,1))
print("Largest readout coefficients:",[(i,round(float(v0[i]),2)) for i in v0.abs().topk(12).indices.tolist()])
print("Largest individual contrast contributions:",[(i,round(float(energy_by_dim[i]),3)) for i in energy_by_dim.topk(12).indices.tolist()])
print("Angle to original effective key readout:",
      torch.nn.functional.cosine_similarity(v0,H0["k.weight"].T@qmean/math.sqrt(768),dim=0).item())
for g in ["invoice_processing/matches_order","invoice_processing/urgency",
          "agent_trace_observability/outcome","customer_service/action"]:
    ix=[j for j,x in enumerate(dev["group"]) if x==g]
    basepred=dev["L"].masked_fill(~dev["mask"],-1e9).argmax(1)
    newpred=(dev["L"]+Cdev).masked_fill(~dev["mask"],-1e9).argmax(1)
    fixed=[j for j in ix if basepred[j]!=dev["gold"][j].argmax() and newpred[j]==dev["gold"][j].argmax()]
    if fixed:
        j=fixed[0]; m=dev["meta"][j]
        print("\nFIX",dev["ids"][j],g)
        print("Options:",m["options"],"centered correction:",center_options(Cdev,dev["mask"])[j,:len(m["options"])].tolist())
        print("State:",json.dumps(byid[dev["ids"][j]]["inputs"]["state"],ensure_ascii=False)[:950])
edit_dir=WORK/"candidate_rank1"
edit_dir.mkdir(exist_ok=True)
torch.save({"u":u,"v":v0,"lambda":lam0,"fit_ids":splits["fit"]},edit_dir/"derivation.pt")
def save_rank1(strength):
    save_edit("k.weight",H0["k.weight"]+strength*deltaK0)
    lognote(f"Saved k.weight = original + {strength:g} * outer(u,v), rank-one key correction; "
            f"u,v in candidate_rank1/derivation.pt, lambda={lam0}. All other tensors original.")
save_rank1(.5)
json.dump({"type":"decide","model":"both","problems":splits["dev"]},open(WORK/"paired_dev.json","w"))
lognote("Causal controls on dev: original "+str(accuracy_stats(dev,dev["L"]))+
        "; static fit-schema offsets "+str(accuracy_stats(dev,dev["L"]+Cdev_static))+
        "; state-dependent rank1 "+str(accuracy_stats(dev,dev["L"]+Cdev))+
        "; within-schema shuffled state corrections "+str(accuracy_stats(dev,dev["L"]+Cshuf))+".")
lognote("## Before pass 6\nExact edit: k.weight = K0 + 0.5*outer(u,v), lambda .001, fit=540 only. "
        "Paired DECIDE on 180 dev IDs in paired_dev.json; first actual dose-response test.")
print("\nSaved edits:",edited())
```
