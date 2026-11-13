I’ll compute four independent, closed-form corrections—one per workflow—from the original option-margin errors. I’ll compare a small, fixed set of regularization strengths on the tuning split only. This tests whether the shared direction is too coarse; no weights will be saved yet.

```python
# Hypothesis: different workflows need different missing option-margin directions, even though the same scorer bottleneck affects all four.
repair_base1_tensors=(repair_b1["first_weight"],repair_b1["first_bias"],repair_b1["last_weight"])
repair_base1_logits=repair_simulate(repair_base1_tensors)
repair_fc=center_options(repair_features[:,:,1024:],repair_mask)
repair_lc=center_options(torch.nan_to_num(repair_logits,neginf=0),repair_mask)
repair_nt=repair_mask.sum(-1).float()
repair_soft=torch.log(repair_gold.clamp_min(.015))
repair_soft=center_options(repair_soft,repair_mask)
repair_hard=center_options(torch.nn.functional.one_hot(repair_target,5).float()*2.5,repair_mask)
repair_targets={"soft":repair_soft-repair_lc,"hard":repair_hard-repair_lc}
repair_wf_dirs={}
repair_wf_logs={}
repair_wf_stats=[]
repair_lambdas=[1e-4,1e-3,1e-2]
for wi,wn in enumerate(repair_workflows):
    sub=repair_fit&(repair_wfids==wi)
    xf=repair_fc[sub]/repair_nt[sub,None,None].sqrt()
    xf=xf[repair_mask[sub]].double()
    kg=xf@xf.T
    evk,uk=torch.linalg.eigh(kg)
    unit=float(kg.trace()/xf.shape[1])
    print("Workflow solve:",wn,tuple(xf.shape),"unit",round(unit,4),"rank",(evk>unit*1e-4).sum().item())
    for tn,targ in repair_targets.items():
        yt=(targ[sub]/repair_nt[sub,None].sqrt())[repair_mask[sub]].double()
        for lm in repair_lambdas:
            dw=(xf.T@(uk@((uk.T@yt)/(evk.clamp_min(0)+lm*unit)))).float()
            repair_wf_dirs[(wi,tn,lm)]=dw
    del xf,kg,evk,uk
for tn in repair_targets:
    for lm in repair_lambdas:
        cor=torch.zeros_like(repair_lc)
        for wi in range(4):
            sub=repair_wfids==wi
            cor[sub]=repair_fc[sub]@repair_wf_dirs[(wi,tn,lm)]
        for st in [.25,.5,.75]:
            ll=(repair_logits+st*cor).masked_fill(~repair_mask,-torch.inf)
            repair_wf_logs[(tn,lm,st)]=ll
            repair_wf_stats.append((tn,lm,st,round(repair_measure(ll,repair_fit)["acc"],4),repair_measure(ll,repair_tune)))
print("Reference best1:",repair_measure(repair_base1_logits,repair_tune),"best2:",repair_measure(repair_current_logits,repair_tune))
for result in repair_wf_stats:
    print(result)
```
