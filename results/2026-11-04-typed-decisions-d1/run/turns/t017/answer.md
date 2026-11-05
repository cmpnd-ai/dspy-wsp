I’ll use exact prompt matching to align all five batches, then compare several regularization strengths. These are separate closed-form solutions, not iterative weight training.

```python
# Hypothesis: the scorer underweights useful contextual directions; a regularized residual readout can recover them.
prompt_ids={json.dumps(json.loads(r["calls"][0]["prompt"]),sort_keys=True):r["id"] for r in records}
Zs=[]; Ls=[]; Ps=[]; Ms=[]; rows=[]; Gs=[]; temps=[]
for b in range(5):
    d=work/f"fwd_{b+2}"; m=json.load(open(d/"meta.json"))
    req=json.load(open(d/"requests.json"))
    ids=[prompt_ids[json.dumps(r,sort_keys=True)] for r in req]
    Zs.append(features(np.load(d/"head_inputs.npy")).numpy())
    L=np.load(d/"option_logits.npy"); P=np.load(d/"probabilities.npy"); M=np.load(d/"option_mask.npy").astype(bool)
    Ls.append(L); Ps.append(P); Ms.append(M)
    for j,r in enumerate(m["rows"]):
        eid=ids[r["request"]]; gold=byid[eid]["labels"]["gold"][r["question"]]["probabilities"]
        g=np.zeros(5)
        for o,opt in enumerate(r["options"]): g[o]=gold[opt]
        Gs.append(g); rows.append((eid,byid[eid]["inputs"]["workflow"],r["question"]))
        l=L[j,M[j]]; p=np.log(P[j,M[j]])
        temps.append(float(((l-l.mean())@(p-p.mean()))/((p-p.mean())@(p-p.mean()))))
Z=np.nan_to_num(np.concatenate(Zs)); L=np.nan_to_num(np.concatenate(Ls)); P=np.nan_to_num(np.concatenate(Ps)); M=np.concatenate(Ms)
G=np.array(Gs); T=np.array(temps); n=M.sum(1)
X=(Z-(Z*M[...,None]).sum(1,keepdims=True)/n[:,None,None])*M[...,None]
lg=np.log(np.maximum(G,0.015))
target=(lg-(lg*M).sum(1,keepdims=True)/n[:,None])*T[:,None]
Lc=L-(L*M).sum(1,keepdims=True)/n[:,None]; Y=(target-Lc)*M
fit=np.array([r[0] in fit_ids for r in rows])
A=torch.tensor(X[fit][M[fit]],dtype=torch.float64); y=torch.tensor(Y[fit][M[fit]],dtype=torch.float64)
gram=A.T@A; rhs=A.T@y
w0=sw["head.scorer.3.weight"].numpy().ravel()
Lmasked=np.where(M,L,-np.inf)
base=Lmasked.argmax(1)==G.argmax(1)
def report(delta, sel):
    scores=L+np.einsum("qod,d->qo",X,delta)
    c=np.where(M,scores,-np.inf).argmax(1)==G.argmax(1)
    diff=c.astype(float)-base.astype(float)
    per=collections.defaultdict(list)
    for i in np.flatnonzero(sel):per[rows[i][0]].append(diff[i])
    ds=np.array([np.mean(v) for v in per.values()])
    return round(c[sel].mean(),4),round(ds.mean(),4),round(ds.std(ddof=1)/np.sqrt(len(ds)),4)
deltas={}
for ridge in [10,100,1000,10000]:
    delta=torch.linalg.solve(gram+ridge*torch.eye(1024,dtype=torch.float64),rhs).float().numpy()
    deltas[ridge]=delta
    print("ridge",ridge,"norm",round(float(np.linalg.norm(delta)),3),"fit",report(delta,fit),"test",report(delta,~fit))
print("Original norm",np.linalg.norm(w0),"temperature",T.min(),T.max())
```
