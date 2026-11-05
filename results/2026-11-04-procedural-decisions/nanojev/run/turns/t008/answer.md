The recorded logits reconstruct exactly from the scalar features plus the set correction. Boolean false and true share the stored feature vector, but only true is scored—so I’ll explicitly preserve false as zero in the calculation. I’ll choose regularization using an internal practice split, leaving the 480-example test subset untouched.

```python
# Hypothesis: ridge-regularized, within-question evidence contrasts repair option priors without changing the backbone.
rows=meta["rows"]
goldvec=[]
Xparts=[]; yparts=[]; reqparts=[]
for i,r in enumerate(rows):
    g=byid[fit_ids[r["request"]]]["labels"]["gold"][r["question"]]["probabilities"]
    p=np.array([g[o] for o in r["options"]],dtype=float)
    goldvec.append(p)
    k=len(p)
    if r["type"]=="noul":
        Xparts.append(H[i,1:2])
        target=np.log(np.clip(p[1],.1,.9)/np.clip(p[0],.1,.9))
        yparts.append(np.array([target-arr["option_logits"][i,1]]))
        reqparts.append([r["request"]])
    else:
        xx=H[i,:k]; zz=arr["option_logits"][i,:k]
        target=3*(p-p.mean())
        Xparts.append((xx-xx.mean(0))/np.sqrt(k))
        yparts.append((target-(zz-zz.mean()))/np.sqrt(k))
        reqparts.append([r["request"]]*k)
X=np.concatenate(Xparts); y=np.concatenate(yparts); reqs=np.concatenate(reqparts)
inner_hold={j for j in range(len(fit_ids)) if j%5==0}
trainmask=np.array([j not in inner_hold for j in reqs])
G=X[trainmask].T@X[trainmask]
eig,U=np.linalg.eigh(G); rhs=U.T@(X[trainmask].T@y[trainmask])
def scores_direction(d, strength=1, subset=None):
    result=collections.defaultdict(list); peritem=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if subset is not None and r["request"] not in subset: continue
        k=len(r["options"])
        z=arr["option_logits"][i,:k].astype(float).copy()
        change=H[i,:k]@d*strength
        if r["type"]=="noul": change[0]=0
        z+=change
        ok=float(np.argmax(z)==np.argmax(goldvec[i]))
        result[r["question"]].append(ok); peritem[r["request"]].append(ok)
    return np.mean([np.mean(v) for v in peritem.values()]),{q:round(np.mean(v),3) for q,v in result.items()}
print("Design",X.shape,"eigenvalues",eig[[0,256,512,768,-1]])
for lam in [1,10,100,1000,10000]:
    d=U@(rhs/(np.maximum(eig,0)+lam))
    print("ridge",lam,"delta norm",round(np.linalg.norm(d),3),"inner heldout",scores_direction(d,subset=inner_hold))
```
