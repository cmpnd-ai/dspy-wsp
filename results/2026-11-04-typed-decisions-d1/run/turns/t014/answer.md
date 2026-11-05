The reconstructed logits match within \(2\times10^{-6}\), so the intervention can be localized to `head.scorer.3.weight`. I’ll derive a regularized, closed-form correction using only the 450 fit cases. Centering within each question removes shared offsets and makes the correction act on option rankings.

```python
# Hypothesis: a directional correction of existing GELU features repairs collapsed option rankings.
Zs=[]; Ls=[]; Ps=[]; Ms=[]; rows=[]; Gs=[]; temps=[]
for b in range(5):
    d=work/f"fwd_{b+2}"
    m=json.load(open(d/"meta.json"))
    H=np.load(d/"head_inputs.npy")
    Zs.append(features(H).numpy())
    L=np.load(d/"option_logits.npy"); P=np.load(d/"probabilities.npy"); M=np.load(d/"option_mask.npy").astype(bool)
    Ls.append(L); Ps.append(P); Ms.append(M)
    for j,r in enumerate(m["rows"]):
        eid=ordered_ids[b*180+r["request"]]
        gold=byid[eid]["labels"]["gold"][r["question"]]["probabilities"]
        g=np.zeros(5)
        for o,opt in enumerate(r["options"]):g[o]=gold[opt]
        Gs.append(g)
        rows.append((eid,byid[eid]["inputs"]["workflow"],r["question"]))
        l=L[j,M[j]]; p=np.log(P[j,M[j]])
        temps.append(float(((l-l.mean())@(p-p.mean()))/((p-p.mean())@(p-p.mean()))))
Z=np.nan_to_num(np.concatenate(Zs)); L=np.nan_to_num(np.concatenate(Ls)); P=np.nan_to_num(np.concatenate(Ps)); M=np.concatenate(Ms)
G=np.array(Gs); T=np.array(temps)
n=M.sum(1)
X=Z-(Z*M[...,None]).sum(1,keepdims=True)/n[:,None,None]
X*=M[...,None]
lg=np.log(np.maximum(G,0.015))
target=(lg-(lg*M).sum(1,keepdims=True)/n[:,None])*T[:,None]
Lc=L-(L*M).sum(1,keepdims=True)/n[:,None]
Y=(target-Lc)*M
fit=np.array([r[0] in fit_ids for r in rows])
A=torch.tensor(X[fit][M[fit]],dtype=torch.float64)
y=torch.tensor(Y[fit][M[fit]],dtype=torch.float64)
gram=A.T@A; rhs=A.T@y
w0=sw["head.scorer.3.weight"].numpy().ravel()
base=L.argmax(1)==G.argmax(1)
def report(delta, sel):
    correct=(L+np.einsum("qod,d->qo",X,delta)).astype(float)
    correct[~M]=-np.inf
    c=correct.argmax(1)==G.argmax(1)
    ds=(c-base)[sel] if False else (c.astype(float)-base.astype(float))[sel]
    ds=ds.reshape(-1,5).mean(1)
    return round(c[sel].mean(),4),round(ds.mean(),4),round(ds.std(ddof=1)/np.sqrt(len(ds)),4)
deltas={}
for ridge in [10,100,1000,10000]:
    delta=torch.linalg.solve(gram+ridge*torch.eye(1024,dtype=torch.float64),rhs).float().numpy()
    deltas[ridge]=delta
    print("ridge",ridge,"norm",np.linalg.norm(delta),"fit",report(delta,fit),"test",report(delta,~fit))
print("Original readout norm",np.linalg.norm(w0),"temperature range",T.min(),T.max())
```
