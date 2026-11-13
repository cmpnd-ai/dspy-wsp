The reconstructed baseline accuracy does not match the program baseline, so I won’t trust those candidate results. The likely cause is request ordering in the read export. I’ll match each read row back to its recorded probabilities and correct the alignment before proceeding.

```python
# Hypothesis: READ-from filters baseline records in file order, not the randomized order of supplied IDs.
def baseline_vector(i, qids=None):
    ans=json.loads(recbyid[i]["calls"][0]["completion"])
    if qids is None:
        qids=list(ans)
    return np.array([p for q in qids for p in ans[q]["probabilities"].values()],dtype=np.float64)
def infer_read_ids(passnum, requested):
    root=WORK/f"fwd_{passnum}"
    meta=json.load(open(root/"meta.json"))["rows"]
    pp=np.load(root/"probabilities.npy")
    nreq=max(m["request"] for m in meta)+1
    matched=[]
    maxerr=0.
    for ri in range(nreq):
        inds=[j for j,m in enumerate(meta) if m["request"]==ri]
        opts=[meta[j]["options"] for j in inds]
        qids=[meta[j]["question"] for j in inds]
        p=np.concatenate([pp[j,:len(meta[j]["options"])] for j in inds])
        cand=[]
        for i in requested:
            ans=json.loads(recbyid[i]["calls"][0]["completion"])
            if list(ans)!=qids:
                continue
            if any(set(ans[q]["probabilities"]) != set(op) for q,op in zip(qids,opts)):
                continue
            pv=np.array([ans[q]["probabilities"][op] for q,ops in zip(qids,opts) for op in ops])
            cand.append((np.max(np.abs(p-pv)),i))
        er,i=min(cand)
        assert er<1e-5,(passnum,ri,er)
        matched.append(i)
        maxerr=max(maxerr,er)
    assert len(set(matched))==len(requested)
    print("Pass",passnum,"matched",len(matched),"max prob error",maxerr,
          "requested order",requested[:3],"actual order",matched[:3])
    return matched
read_ids={}
for j,(_,ids) in zip([2,3,4,5],read_plan[:4]):
    read_ids[j]=infer_read_ids(j,ids)
fit=concat_data([load_read(j,read_ids[j]) for j in [2,3,4]])
dev=load_read(5,read_ids[5])
for data in [fit,dev]:
    data["Q"]=head_q(data["d"])
qmean=fit["Q"].mean(0)
u=qmean/qmean.norm()
for data in [fit,dev]:
    data["g"]=data["Q"]@u/math.sqrt(768)
print("Corrected original fit/dev",accuracy_stats(fit,fit["L"]),accuracy_stats(dev,dev["L"]))
print("From baseline records fit/dev",*[np.mean([r["ok"] for r in rows if id_split[r["id"]]==s]) for s in ["fit","dev"]])
lognote("IMPORTANT alignment correction: READ-from preserves baseline-file order, not supplied randomized ID order. "
        "Initial ridge grid from Python run 8 used misaligned labels and is INVALID/discarded; no edits were saved. "
        "Matched every request to baseline completion probabilities (max error <1e-5); verified one-to-one mapping "
        "and corrected labels. All subsequent analyses use read_ids matched from actual probabilities.")
json.dump(read_ids,open(WORK/"read_id_alignment.json","w"))
del ridge_vs,results_ridge,X,Xfit3,Xdev3,gram,rhs,evals,evecs,projrhs
```
