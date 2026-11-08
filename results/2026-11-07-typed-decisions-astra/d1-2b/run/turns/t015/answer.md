I’ll check that the untouched coordinates differ only by a common normalization factor. If that holds, the intervention cleanly identifies the last-write contribution and supports a controlled output-direction edit.
```python
# Hypothesis: untouched coordinates obey a single RMS scale ratio, validating causal recovery of the perturbed residual contribution.
def read_arrays(k):
    folder=WORK/f"fwd_{k}"
    meta=json.loads((folder/"meta.json").read_text())
    reqs=json.loads((folder/"requests.json").read_text())
    sig=lambda obj:json.dumps(obj,sort_keys=True,separators=(",",":"))
    lookup={sig(json.loads(r["calls"][0]["prompt"])):r["id"] for r in baseline}
    ids=[lookup.get(sig(req)) for req in reqs]
    h=np.load(folder/"head_inputs.npy")[:,0,:].copy()
    mask=np.load(folder/"option_mask.npy")
    erows=np.nan_to_num(np.load(folder/"option_rows.npy"),nan=0.0)
    logits=np.nan_to_num(np.load(folder/"option_logits.npy"),nan=-1e8)
    probs=np.nan_to_num(np.load(folder/"probabilities.npy"),nan=0.0)
    gold=np.zeros_like(probs)
    row_ids=[]; keys=[]
    for i,row in enumerate(meta["rows"]):
        eid=ids[row["request"]]
        row_ids.append(eid)
        if eid is not None:
            keys.append((by_id[eid]["inputs"]["workflow"],row["question"]))
            gs=by_id[eid]["labels"]["gold"][row["question"]]["probabilities"]
            for j,opt in enumerate(row["options"]):
                gold[i,j]=gs[opt]
        else:
            keys.append((None,row["question"]))
    return dict(meta=meta,reqs=reqs,ids=ids,h=h,mask=mask,e=erows,l=logits,p=probs,gold=gold,
                row_ids=row_ids,keys=keys)
data={}
for group,k in [("fit_a",2),("fit_b",3),("dev",4)]:
    data[group]=read_arrays(k)
def infer_subset(orig, pert, changed):
    unchanged=np.setdiff1d(np.arange(2048),changed)
    h=orig["h"].astype(np.float64); hp=pert["h"].astype(np.float64)
    scale=(hp[:,unchanged]*h[:,unchanged]).sum(1)/(h[:,unchanged]**2).sum(1)
    anchor_error=np.sqrt(np.mean((hp[:,unchanged]-scale[:,None]*h[:,unchanged])**2))
    c=(hp[:,changed]/scale[:,None]-h[:,changed])/probe_delta/gn.numpy()[changed]
    return c.astype(np.float32),scale,anchor_error
probe5=read_arrays(5)
ceven,scale5,err5=infer_subset(data["fit_a"],probe5,even_idx)
print("Request match/order:",data["fit_a"]["ids"][:3], "same in probe:",data["fit_a"]["ids"]==probe5["ids"])
print("Anchor RMS error:",err5,"scale quantiles",np.quantile(scale5,[0,.5,1]))
print("Mean baseline accuracy:",np.mean(data["fit_a"]["p"].argmax(1)==data["fit_a"]["gold"].argmax(1)))
print("Even-gain probe accuracy:",np.mean(probe5["p"].argmax(1)==data["fit_a"]["gold"].argmax(1)))
print("Mean absolute option-logit change:",np.mean(np.abs((probe5["l"]-data["fit_a"]["l"])[data["fit_a"]["mask"]])))
note(f"Pass 5 causal check: untouched-coordinate RMS error {err5:.3g}; final-scale ratio median {np.median(scale5):.5f}. Probe changes option logits (mean |delta|={np.mean(np.abs((probe5['l']-data['fit_a']['l'])[data['fit_a']['mask']])):.4f}).\n\nPlanned pass 6: READ edited fit_b (180 IDs in probe_even_fit_b.json). Exact edit unchanged: post_ffn_norm.23 even gains x1.5; odd original. No w2 edit.")
```
