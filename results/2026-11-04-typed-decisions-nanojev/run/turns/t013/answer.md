The read tool preserves the order of the earlier baseline records, not the requested ID list. I’ll correct that mapping before fitting so every representation is paired with the right gold distribution.

```python
# Hypothesis: correct request-to-case alignment is essential for a meaningful closed-form readout repair.
for b,chunk in zip(batches,fit_chunks):
    ordered=[r["id"] for r in records if r["id"] in set(chunk)]
    b["ids"]=[ordered[r["request"]] for r in b["rows"]]
designs=[design(b) for b in batches]
X=np.concatenate([d[0] for d in designs[:2]])
y=np.concatenate([d[1] for d in designs[:2]])
G=X.T@X; rhs=X.T@y
evals,U=np.linalg.eigh(G)
print("Design",X.shape,"Gram eig quantiles",np.quantile(evals,[0,.25,.5,.75,1]))
def eval_dw(dw, groups, strength=1):
    hits=[]; bs=[]
    for h,z,p,r in groups:
        zz=z+strength*(h@dw)
        pp=np.exp(zz-zz.max()); pp/=pp.sum()
        hits.append(int(pp.argmax()==p.argmax()))
        bs.append(np.mean((pp-p)**2))
    return np.mean(hits),np.mean(bs)
for lam in [0.1,1,10,100,1000]:
    dw=U@((U.T@rhs)/(evals+lam))
    print("lambda",lam,"norm",round(np.linalg.norm(dw),3),
          "fit",eval_dw(dw,designs[0][2]),"internal holdout",eval_dw(dw,designs[2][2]))
print("Set correction within-choice spread", np.quantile([
    np.ptp(batches[0]["set_delta"][i,:len(r["options"])])
    for i,r in enumerate(batches[0]["rows"]) if r["type"]=="choice"],[0,.5,.9,1]))
```
