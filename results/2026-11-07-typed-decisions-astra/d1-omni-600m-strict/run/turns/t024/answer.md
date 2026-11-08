The counterfactual test found a specific failure: making an invoice overdue slightly *lowers* its elevated-versus-routine margin, and an expiring discount barely changes it. Yet a payment-sensitive direction is reproducible in the head features. I’ll make that direction more selective by removing dominant variation from unrelated questions before testing a direct scorer repair.
```python
# Hypothesis: removing unrelated option-variation directions will let the scorer read payment evidence with limited cross-workflow effects.
# Build a label-free nuisance subspace from unrelated question contrasts in discovery A.
nuisance_samples=[]
for r in rowinfo2:
    if (r["wf"],r["q"])==("invoice_processing","urgency"):
        continue
    mat=U2[r["i"],:len(r["opts"])]
    nuisance_samples.extend(mat-mat.mean(0))
NS=np.stack(nuisance_samples).astype(np.float64)
NS_res=NS-(NS@Qn)@Qn.T
_,sn,vhn=np.linalg.svd(NS_res,full_matrices=False)
protected_directions={}
for rank in [0,16,32,64]:
    qq=np.concatenate([Qn,vhn[:rank].T],axis=1)
    d=causal_mean-qq@(qq.T@causal_mean)
    d/=np.linalg.norm(d)
    protected_directions[rank]=d
    print("rank",rank,"remaining causal norm",float(causal_mean@d))
    for flag in [False,True]:
        rs=[r for r in rowinfo3 if r["wf"]=="invoice_processing" and r["q"]=="urgency" and
            (by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"]<=3)==flag]
        print(" B near",flag,"mean",round(float(np.mean([(U3[r["i"],2]-U3[r["i"],1])@d for r in rs])),5))
    spans=[np.ptp(U3[r["i"],:len(r["opts"])]@d) for r in rowinfo3 if (r["wf"],r["q"])!=("invoice_processing","urgency")]
    print(" B unrelated mean/95% span",np.round(np.quantile(spans,[.5,.95]),5),"mean",round(float(np.mean(spans)),5))
```
