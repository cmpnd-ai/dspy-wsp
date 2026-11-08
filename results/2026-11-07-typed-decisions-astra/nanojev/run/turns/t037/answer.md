The set-output experiments add only about half a percentage point on development data, so I won’t add that extra edit. I’ll check one final closed-form alternative that emphasizes distinctions among plausible options, then collect the remaining practice features for a larger refit using a frozen recipe.
```python
# Hypothesis: a one-shot Fisher-weighted contrast solve can focus the readout on plausible option distinctions.
# No iterative updates: each candidate is an independent closed-form ridge solve from original features/logits.
fisher_candidates={};fisher_results=[]
for name,prob,target in [
    ("gold_log",.9*G+.1*mask/counts[:,None],np.log(np.maximum(G,.01))),
    ("mixed_log",.9*(.5*G+.5*onehot)+.1*mask/counts[:,None],np.log(np.maximum(.5*G+.5*onehot,.01))),
    ("top_margin",.25*mask/counts[:,None]+.75*onehot,3*(.5*G+.5*onehot)),
]:
    prob=prob*mask;prob/=prob.sum(1,keepdims=True)
    mu=(X*prob[...,None]).sum(1,keepdims=True)
    XD=(X-mu)*np.sqrt(prob[...,None])
    res=target-L0
    ry=(res-(res*prob).sum(1,keepdims=True))*np.sqrt(prob)
    xd=torch.from_numpy(XD[fit_rows][mask[fit_rows]])
    yd=torch.from_numpy(ry[fit_rows][mask[fit_rows]])
    gg=xd.T@xd; rr=xd.T@yd
    ee,vv=torch.linalg.eigh(gg); bb=vv.T@rr
    for lam in [.003,.01,.03]:
        dw=(vv@(bb/(ee+lam))).numpy()
        fisher_candidates[(name,lam)]=dw
        for strength in [.75,1.0]:
            pred=L0+strength*(X@dw)
            fisher_results.append((metric_logits(pred,dev_rows)[0],metric_logits(pred,dev_rows)[1],
                                   metric_logits(pred,fit_rows)[0],name,lam,strength,np.linalg.norm(dw)))
print("Weighted-contrast candidates: dev accuracy, Brier, fit accuracy, target, ridge, strength, norm")
for item in sorted(fisher_results,reverse=True)[:12]:
    print(tuple(round(x,4) if isinstance(x,(float,np.floating)) else x for x in item))
note("Set head residual/joint fits yield at most dev0.605333 vs mixed scalar0.600000, too small to justify a second component. No set edits saved. Testing prespecified one-shot Fisher/top-margin weighted contrasts using fit450/dev150 only.")
read15_ids=splits["test"][:200]
(WORK/"read15_request.json").write_text(json.dumps({"type":"read","model":"original","from":{"pass":1,"ids":read15_ids}}))
note("Planned forward15: original READ first200 test IDs, exact read15_request.json,1000 questions. Current edit unchanged best2 scalar.weight=w_original+.75*d_mix600; original mode ignores it. These300 will be retired as independent data when used for full900 refit; prior independent practice results remain valid for frozen earlier candidates. New recipe selection remains fit450/dev150 only.")
```
