The shared correction helps most workflows, but it still mixes incompatible option roles—for example, “critical” urgency and “high” risk are not interchangeable. I’ll test whether question-specific linear reads of the existing field vectors can recover more of the missing evidence. These are closed-form diagnostic fits; no new weights will be saved yet.

```python
# Hypothesis: the head retains state evidence in field vectors, but the shared scorer cannot read it differently for each question and option.
field_features = D["features"][:,0,:1024].double()
conditional_candidates = {}
conditional_table = []
target_sets = {"soft":target_log,"mixed":alt_targets["mixed"],"hard":alt_targets["hard_margin"]}
qdata={}
for qkey in qkeys:
    ix=torch.tensor([k==qkey for k in D["qkey"]])
    tr=ix & D["isfit"]
    te=ix & D["istune"]
    x=field_features[tr]
    mu=x.mean(0)
    xc=x-mu
    gram=xc@xc.T
    ev,U=torch.linalg.eigh(gram)
    scale=gram.diag().mean().item()
    qdata[qkey] = dict(ix=ix,tr=tr,te=te,mu=mu,xc=xc,ev=ev,U=U,scale=scale)
for tname,T in target_sets.items():
    residual=T-center_options(D["logits"],D["mask"])
    for ridge in [1e-4,1e-3,1e-2,.1,1.]:
        correction=torch.zeros_like(D["logits"])
        param={}
        for qkey,qd in qdata.items():
            yy=residual[qd["tr"]].double()
            my=yy.mean(0)
            coef=qd["xc"].T @ (qd["U"] @ ((qd["U"].T@(yy-my))/(qd["ev"][:,None]+ridge*qd["scale"])))
            intercept=my-qd["mu"]@coef
            correction[qd["ix"]]=(field_features[qd["ix"]]@coef+intercept).float()
            param[qkey]=(coef.float(),intercept.float())
        conditional_candidates[tname,ridge]={"correction":correction,"param":param}
        for strength in [.5,1.]:
            L=D["logits"]+strength*correction
            ff,tt=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
            conditional_table.append((tname,ridge,strength,ff["accuracy"],tt["accuracy"],tt["gain"],tt["se"]))
print("target ridge strength fit tune gain paired_se")
for tname,r,s,f,t,g,se in conditional_table:
    print(tname,r,s,*[round(v,4) for v in [f,t,g,se]])
note("## Conditional readout diagnostic\nFitting each question's missing option margins as a direct ridge read of its existing field vector (125 fit records/question). This tests whether evidence is present before final scoring. Each solve is closed-form; no iterative loss fitting. Three predetermined targets × five regularizers, tune only, with all 300 reserved test records still excluded from fitting and selection.")
```
