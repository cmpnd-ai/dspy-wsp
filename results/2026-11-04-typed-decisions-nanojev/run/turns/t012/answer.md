I’ll derive a closed-form correction to the scalar weights using centered log-probability targets. Centering makes the edit address relative option preferences rather than globally sharpening distributions. I’ll compare regularization strengths using only the 600-case fitting split.

```python
# Hypothesis: a regularized correction of misread EOS directions can restore state-sensitive option rankings.
def read_batch(passno, batchids):
    d=work/f"fwd_{passno}"
    a={n:np.load(d/f"{n}.npy") for n in ["option_mask","option_logits","head_inputs","set_delta","probabilities"]}
    rows=json.load(open(d/"meta.json"))["rows"]
    a["rows"]=rows
    a["ids"]=[batchids[r["request"]] for r in rows]
    return a
batches=[read_batch(k+2,fit_chunks[k]) for k in range(3)]
def design(batch):
    xs=[]; ys=[]; groups=[]
    for i,r in enumerate(batch["rows"]):
        n=len(r["options"]); h=batch["head_inputs"][i,:n].astype(np.float64).copy()
        z=batch["option_logits"][i,:n].astype(np.float64)
        gp=byid[batch["ids"][i]]["labels"]["gold"][r["question"]]["probabilities"]
        p=np.array([gp[o] for o in r["options"]])
        t=np.log(np.maximum(p,0.015))
        if r["type"]=="noul": h[0]=0
        h-=h.mean(axis=0)
        y=(t-t.mean())-(z-z.mean())
        xs.append(h/np.sqrt(n)); ys.append(y/np.sqrt(n))
        groups.append((h,z,p,r))
    return np.concatenate(xs),np.concatenate(ys),groups
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
