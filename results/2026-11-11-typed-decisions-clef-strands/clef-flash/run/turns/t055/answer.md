Many unused scorer neurons sit below GELU’s active range and contribute almost nothing to option margins. I’ll test whether recentering those neurons exposes useful nonlinear evidence. The incoming directions stay fixed; the readout is solved once in closed form against the original errors, with fit and tuning data kept separate.


```python
# Hypothesis: reactivating suppressed original scorer directions adds nonlinear evidence beyond the validated linear correction.
repair_dead_ix=torch.where(repair_dormant & (repair_mu0<0))[0]
repair_recenter_b=repair_b0.clone()
repair_recenter_b[repair_dead_ix]=(repair_b0[repair_dead_ix]-repair_mu0[repair_dead_ix]).bfloat16().float()
repair_react=F.gelu(repair_pre0[:,:,repair_dead_ix]+(repair_recenter_b-repair_b0)[repair_dead_ix])
repair_react_c=center_options(repair_react,repair_mask)
repair_nonlinear_scale=float(repair_fc[repair_fit][repair_mask[repair_fit]].square().mean().sqrt()/
                             repair_react_c[repair_fit][repair_mask[repair_fit]].square().mean().sqrt())
print("Reactivated neurons:",len(repair_dead_ix),"feature scale",repair_nonlinear_scale)
repair_aug_fc=torch.cat([repair_fc,repair_react_c*repair_nonlinear_scale],dim=-1)
xf=(repair_aug_fc[repair_fit]/repair_nt[repair_fit,None,None].sqrt())[repair_mask[repair_fit]].double()
ag=xf.T@xf
ae,au=torch.linalg.eigh(ag)
aunit=float(ag.trace()/ag.shape[0])
repair_react_candidates={}
repair_react_stats=[]
for tn in ["soft","hard"]:
    yt=(repair_targets[tn][repair_fit]/repair_nt[repair_fit,None].sqrt())[repair_mask[repair_fit]].double()
    arhs=xf.T@yt
    for lm in [1e-4,1e-3,1e-2]:
        dw=(au@((au.T@arhs)/(ae.clamp_min(0)+lm*aunit))).float()
        repair_react_candidates[(tn,lm)]=dw
        corr=repair_aug_fc@dw
        for st in [.5,.75]:
            ll=(repair_logits+st*corr).masked_fill(~repair_mask,-torch.inf)
            repair_react_stats.append((tn,lm,st,round(repair_measure(ll,repair_fit)["acc"],4),repair_measure(ll,repair_tune)))
for rr in repair_react_stats: print(rr)
del xf,ag,ae,au
```
