The ordering check found a label alignment error: reads are returned in baseline order. I’ll correct that before deriving any edit. No weights have been changed.

```python
# Hypothesis: correctly aligned labels reveal whether the scalar head misses usable rule evidence.
fit_ids_requested=fit_ids.copy()
fit_ids=fit_order
goldvec=[]; Xparts=[]; yparts=[]; reqparts=[]
for i,r in enumerate(rows):
    g=byid[fit_ids[r["request"]]]["labels"]["gold"][r["question"]]["probabilities"]
    p=np.array([g[o] for o in r["options"]],dtype=float); goldvec.append(p); k=len(p)
    if r["type"]=="noul":
        Xparts.append(H[i,1:2])
        target=np.log(np.clip(p[1],.1,.9)/np.clip(p[0],.1,.9))
        yparts.append(np.array([target-arr["option_logits"][i,1]])); reqparts.append([r["request"]])
    else:
        xx=H[i,:k]; zz=arr["option_logits"][i,:k]
        Xparts.append((xx-xx.mean(0))/np.sqrt(k))
        yparts.append((3*(p-p.mean())-(zz-zz.mean()))/np.sqrt(k)); reqparts.append([r["request"]]*k)
X=np.concatenate(Xparts); y=np.concatenate(yparts); reqs=np.concatenate(reqparts)
inner_hold={j for j in range(len(fit_ids)) if j%5==0}
trainmask=np.array([j not in inner_hold for j in reqs])
eig,U=np.linalg.eigh(X[trainmask].T@X[trainmask]); rhs=U.T@(X[trainmask].T@y[trainmask])
print("Original inner heldout",scores_direction(np.zeros(1024),subset=inner_hold))
for lam in [.01,.1,1,10,100,1000]:
    d=U@(rhs/(np.maximum(eig,0)+lam))
    print(lam,"norm",round(np.linalg.norm(d),3),scores_direction(d,subset=inner_hold))
```
