I’ll rebuild the three targeted corrections with the same formulas and settings, using all 900 practice records. Schema-identity gates will be fitted only to question/option identities—not answer labels—and bfloat16 offset compensation will be computed directly from the resulting features.


```python
# Hypothesis: fixed-design targeted readouts and schema gates become more reliable with all practice records.
full_field=FULL["features"][:,0,:1024]
full_soft=center_options(FULL["gold"].clamp_min(.015).log(),FULL["mask"])
full_field_params={}
full_field_correction=torch.zeros_like(FULL["lc"])
for qk in target_qkeys:
    ix=torch.tensor([x==qk for x in FULL["qkey"]])
    ff=full_field[ix].double()
    xm=ff.mean(0); xc=ff-xm
    yy=(full_soft-FULL["lc"])[ix].double()
    ym=yy.mean(0); yc=yy-ym
    gram=xc@xc.T
    lam=.01*float(gram.trace()/len(ff))
    coef=(xc.T@torch.linalg.solve(gram+lam*torch.eye(len(ff),dtype=torch.float64),yc)).float()
    intercept=(ym-xm@coef.double()).float()
    full_field_params[qk]=(coef,intercept)
    full_field_correction[ix]=full_field[ix]@coef+intercept
full_class_ids=torch.full(FULL["mask"].shape,-1,dtype=torch.long)
for i,(qk,rr) in enumerate(zip(FULL["qkey"],FULL["meta"])):
    for j,oo in enumerate(rr["options"]):
        full_class_ids[i,j]=class_index[(qk,str(oo))]
full_gfeatures=FULL["features"][:,:,:2048]
ggx=full_gfeatures[FULL["mask"]].double()
ggm=ggx.mean(0); ggxc=ggx-ggm
ggy=F.one_hot(full_class_ids[FULL["mask"]],len(option_classes)).double()[:,selected_classes]
ggym=ggy.mean(0)
ggg=ggxc.T@ggxc
ggl=1e-5*float(ggg.trace()/2048)
full_gcoef=torch.linalg.solve(ggg+ggl*torch.eye(2048,dtype=torch.float64),ggxc.T@(ggy-ggym)).float()
full_gbias=(ggym-ggm@full_gcoef.double()).float()
full_gscores=full_gfeatures@full_gcoef+full_gbias
print("Gate positive min / negative max:",min(float(full_gscores[:,:,kk][full_class_ids==cc].min()) for kk,cc in enumerate(selected_classes)),
      max(float(full_gscores[:,:,kk][FULL["mask"]&(full_class_ids!=cc)].max()) for kk,cc in enumerate(selected_classes)))
del ggx,ggxc,ggy,ggg
full_expert_rows={}
full_shared_row=full_linear[0][repair_pair[0]]*repair_gate*full_gain
for kk,cc in enumerate(selected_classes):
    qk,oid=option_classes[cc]
    representative=next(rr for qq,rr in zip(FULL["qkey"],FULL["meta"]) if qq==qk)
    jj=representative["options"].index(oid)
    coef,intercept=full_field_params[qk]
    erow=-full_shared_row.clone()
    erow[:1024]+=coef[:,jj]
    full_expert_rows[cc]=(erow/(repair_gate*full_gain),float(intercept[jj]/(repair_gate*full_gain)))
full_targeted=[a.clone() for a in full_linear]
for kk,cc in enumerate(selected_classes):
    erow,ebias=full_expert_rows[cc]
    grow=torch.cat([full_gcoef[:,kk],torch.zeros(2048)])*32
    gb=(float(full_gbias[kk])-.5)*32
    for sign,neuron in zip([1.,-1.],repair_b2["neurons"][2+2*kk:4+2*kk]):
        full_targeted[0][neuron]=grow+sign*.5*erow
        full_targeted[1][neuron]=gb+sign*.5*ebias
        full_targeted[2][neuron]=sign*full_gain
full_targeted=[a.bfloat16().float() for a in full_targeted]
full_offsets=[]
for kk,cc in enumerate(selected_classes):
    ip,im=repair_b2["neurons"][2+2*kk:4+2*kk]
    sel=full_class_ids==cc
    ff=FULL["features"][sel]
    erow,ebias=full_expert_rows[cc]
    wanted=ff@erow+ebias
    actual=ff@(full_targeted[0][ip]-full_targeted[0][im])+full_targeted[1][ip]-full_targeted[1][im]
    diff=float((wanted-actual).mean())
    full_targeted[1][ip]+=diff/2
    full_targeted[1][im]-=diff/2
    full_offsets.append(diff)
full_targeted=tuple(a.bfloat16().float() for a in full_targeted)
full_targeted_logits=full_simulate(full_targeted)
print("Offset range:",min(full_offsets),max(full_offsets))
print("In-sample practice only:",full_measure(full_targeted_logits))
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],full_targeted):
    save_edit(name,arr.reshape(SHAPES[name]))
torch.save({"tensors":full_targeted,"field_params":full_field_params,"gate_coef":full_gcoef,
            "gate_bias":full_gbias,"offsets":full_offsets,"neurons":repair_b2["neurons"]},WORK/"full_targeted_candidate.pt")
with open(WORK/"notes.md","a") as f:
    f.write("\n## Full-practice fixed-design targeted candidate\nSame 24 neurons as best_2. Shared hard-margin ridge 1e-4, half strength; three selected field readouts soft ridge .01; schema gates ridge 1e-5, threshold .5, strength32; direct bfloat16 offset compensation. All refit once on 900 practice. In-sample practice: "+str(full_measure(full_targeted_logits))+". Not an independent test.\n")
    f.write("\n## Planned pass 15\nExact edits in full_targeted_candidate.pt: 24 scorer rows/biases/output entries, same three tensors only. DECIDE both on 300 hidden validation items. No backbone/routing changes; best_2 saved as fallback.\n")
```
