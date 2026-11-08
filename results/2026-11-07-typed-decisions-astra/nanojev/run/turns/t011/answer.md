The tool returns rows in source-file order, not the requested ID order. I’ll fix the alignment check before using any labels; this prevents a silent mismatch between hidden features and targets.
```python
# Hypothesis: correct source-order alignment is essential to isolate a real readout defect rather than fit mismatched labels.
def load_read_dataset(k,ids):
    folder=WORK/f"fwd_{k}"
    meta=json.loads((folder/"meta.json").read_text())
    arr={p.stem:np.array(np.load(p)) for p in folder.glob("*.npy")}
    order_source=[str(r["id"]) for r in baseline if str(r["id"]) in set(ids)]
    def err(order):
        out=[]
        for j,m in enumerate(meta["rows"]):
            pp=completion_probs(baseline_by_id[order[m["request"]]])
            if m["question"] not in pp: return float("inf")
            pp=pp[m["question"]]
            if any(str(key) not in pp for key in m["options"]): return float("inf")
            out.append(max(abs(arr["probabilities"][j,o]-pp[str(key)]) for o,key in enumerate(m["options"])))
        return max(out)
    errors={"source":err(order_source),"requested":err(ids)}
    order=order_source if errors["source"]<errors["requested"] else ids
    assert min(errors.values())<1e-5,errors
    rows_local=[]
    split_for_id={pid:s for s,ss in splits.items() for pid in ss}
    for j,m in enumerate(meta["rows"]):
        pid=order[m["request"]]; ex=by_id[pid]
        rows_local.append({**m,"id":pid,"workflow":ex["inputs"]["workflow"],"split":split_for_id[pid]})
    print("Read",k,"alignment max errors",errors)
    return rows_local,arr
reads={k:load_read_dataset(k,ids) for k,ids in read_pass_ids.items()}
rows=sum([reads[k][0] for k in [2,3,4]],[])
A={name:np.concatenate([reads[k][1][name] for k in [2,3,4]],axis=0) for name in reads[2][1]}
X=np.nan_to_num(A["head_inputs"]).astype(np.float64)
mask=np.asarray(A["option_mask"],bool)
types=np.array([m["type"] for m in rows])
X[types=="noul",0,:]=0
G=np.zeros(mask.shape,dtype=np.float64)
for j,m in enumerate(rows):
    gold=by_id[m["id"]]["labels"]["gold"][m["question"]]["probabilities"]
    G[j,:len(m["options"])]=[gold[str(o)] for o in m["options"]]
G/=G.sum(axis=1,keepdims=True)
fit_rows=np.array([m["split"]=="fit" for m in rows])
dev_rows=np.array([m["split"]=="dev" for m in rows])
def metric_logits(logits,selection):
    valid_logits=np.where(mask,logits,-1e20)
    preds=np.argmax(valid_logits,axis=-1)
    correct=preds==G.argmax(axis=-1)
    p=np.exp(valid_logits-valid_logits.max(axis=-1,keepdims=True)); p/=p.sum(axis=-1,keepdims=True)
    brier=np.sum((p-G)**2,axis=-1)
    return float(correct[selection].mean()),float(brier[selection].mean())
raw=X@head0["scalar.weight"].numpy().reshape(-1)+head0["scalar.bias"].item()
raw[types=="noul",0]=0
print("Baseline vs no set attention fit/dev:",
      metric_logits(A["option_logits"],fit_rows),metric_logits(raw,fit_rows),
      metric_logits(A["option_logits"],dev_rows),metric_logits(raw,dev_rows))
counts=mask.sum(axis=1)
Xc=(X-(X*mask[...,None]).sum(axis=1,keepdims=True)/counts[:,None,None])*mask[...,None]
logG=np.log(np.maximum(G,0.01))
Y=(logG-(logG*mask).sum(axis=1,keepdims=True)/counts[:,None])*mask
L0=np.nan_to_num(A["option_logits"]).astype(np.float64)
L0c=(L0-(L0*mask).sum(axis=1,keepdims=True)/counts[:,None])*mask
RW=1/np.sqrt(counts)
D=(Xc[fit_rows]*RW[fit_rows,None,None])[mask[fit_rows]]
y=((Y-L0c)[fit_rows]*RW[fit_rows,None])[mask[fit_rows]]
Dt=torch.from_numpy(D); yt=torch.from_numpy(y)
gram=Dt.T@Dt
rhs=Dt.T@yt
eigvals,eigvecs=torch.linalg.eigh(gram)
projections=eigvecs.T@rhs
print("Fit rows/features:",D.shape,"eigenvalue quantiles:",np.quantile(eigvals.numpy(),[0,.1,.5,.9,.99,1]).round(5))
solutions={}
for lam in [0.01,0.03,0.1,0.3,1,3,10,30,100,300]:
    delta=(eigvecs@(projections/(eigvals+lam))).numpy()
    solutions[lam]=delta
    candidate=L0+X@delta
    candidate[types=="noul",0]=0
    print("ridge",lam,"delta_norm",round(float(np.linalg.norm(delta)),3),
          "fit",tuple(round(v,4) for v in metric_logits(candidate,fit_rows)),
          "dev",tuple(round(v,4) for v in metric_logits(candidate,dev_rows)))
note("Read alignment now verified: source-file order, max probability error zero. Scalar correction design: delta=(D'D+lambda I)^-1 D'(center(log gold)-center(original logits)); center features per question, fixed false feature zero, equal question weight. Fit450 only; dev150 used for ridge/strength choice. Original set-attention retained for preliminary cached evaluation; exact recomputation next. No edits saved yet.")
```
