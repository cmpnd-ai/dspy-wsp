Both probes isolated the last write with numerical precision. I’ll restore the original norm and derive a repair confined to nine directions: the contrasts among choice tokens, score tokens, and yes/no. A single closed-form ridge solve will map the observed last-write features to the needed contrast corrections; development data will choose regularization and strength.
```python
# Hypothesis: state-dependent evidence in the final FFN write can be redirected into nine option-contrast directions without changing protected tensors.
drop_edit(POST_NAME)
for group,ke,ko in [("fit_b",6,9),("dev",7,10)]:
    pe,po=read_arrays(ke),read_arrays(ko)
    ce,se,ee=infer_subset(data[group],pe,even_idx)
    co,so,eo=infer_subset(data[group],po,odd_idx)
    c=np.zeros_like(data[group]["h"])
    c[:,even_idx]=ce; c[:,odd_idx]=co
    data[group]["c"]=c
    data[group]["u"]=c/gpost.numpy()
    print(group,"anchor errors",ee,eo)
def combine_data(a,b):
    out={}
    for k in ["h","mask","e","l","p","gold","c","u"]:
        out[k]=np.concatenate([a[k],b[k]],axis=0)
    for k in ["row_ids","keys"]:
        out[k]=a[k]+b[k]
    out["meta"]={"rows":a["meta"]["rows"]+b["meta"]["rows"]}
    return out
fit=combine_data(data["fit_a"],data["fit_b"])
dev=data["dev"]
gn_np=gn.numpy().copy(); gp_np=gpost.numpy().copy()
families={}
for i,row in enumerate(fit["meta"]["rows"]):
    typ=row["type"]
    if typ not in families or len(row["options"])>len(families[typ]):
        families[typ]=fit["e"][i,:len(row["options"])].copy()
D=np.concatenate([(E-E.mean(axis=0))*gn_np*gp_np for E in families.values()],axis=0)
_,sv,vt=np.linalg.svd(D.astype(np.float64),full_matrices=False)
rank=int((sv>sv[0]*1e-5).sum())
B=vt[:rank].T.astype(np.float32)
print("Option contrast rank:",rank,"singular values",sv.round(5))
def temp_for_data(d):
    temps=[]
    for l,p,m in zip(d["l"],d["p"],d["mask"]):
        lc=l[m]-l[m].mean(); lp=np.log(np.maximum(p[m],1e-30)); lp-=lp.mean()
        temps.append(float(lc@lp/(lp@lp)) if lp@lp>1e-10 else 1.3374)
    return np.array(temps)
for d in [fit,dev]:
    d["temp"]=temp_for_data(d)
print("Temperature quantiles:",np.quantile(fit["temp"],[0,.1,.5,.9,1]))
def contrast_targets(d, floor=0.015):
    n=len(d["h"]); targets=np.zeros((n,rank),np.float32); residuals=[]
    for i in range(n):
        m=d["mask"][i]; E=d["e"][i,m].astype(np.float64)
        u=d["u"][i].astype(np.float64); t=d["h"][i].astype(np.float64)/gn_np
        # Jacobian only of the observed final write's RMS normalization and tied readout.
        dwrite=B.astype(np.float64)-u[:,None]*(u@B)[None,:]/(u@u)
        dt=gp_np[:,None]*dwrite
        dh=gn_np[:,None]*(dt-t[:,None]*(t@dt)[None,:]/2048)
        J=E@dh
        J-=J.mean(axis=0)
        gold=d["gold"][i,m].astype(np.float64)
        target=d["temp"][i]*np.log(np.maximum(gold,floor))
        target-=target.mean()
        orig=d["l"][i,m].astype(np.float64); orig-=orig.mean()
        delta=target-orig
        coef=np.linalg.pinv(J,rcond=1e-6)@delta
        targets[i]=coef
        residuals.append(np.linalg.norm(J@coef-delta))
    return targets,np.array(residuals)
Y,linres=contrast_targets(fit)
print("Target correction norm quantiles:",np.quantile(np.linalg.norm(Y,axis=1),[0,.5,.9,1]),"Jacobian fit max residual",linres.max())
U=torch.from_numpy(fit["u"].copy()).double()
Yt=torch.from_numpy(Y).double()
tstart=time.time()
Gram=U@U.T
evals,evecs=torch.linalg.eigh(Gram)
print("Gram eig time",time.time()-tstart,"quantiles",np.quantile(evals.numpy(),[0,.1,.5,.9,.99,1]))
note("Diagnostic norm edits removed. Current edit: none. Final-write features recovered for all fit and dev records (anchor errors ~5e-8). Option-aware pre-gain contrast subspace has rank 9. Targets: gold log-probability contrasts at observed calibration temperatures (probability floor .015), minus baseline contrasts; inverted through analytic final-write/final-RMS/readout Jacobian. Coefficients will be solved once per ridge value, in closed form, on fit=360 records only. No autograd, optimization, or backbone simulation.")
```
