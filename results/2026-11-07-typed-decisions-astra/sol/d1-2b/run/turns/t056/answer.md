I’ll now measure how often the edit changes the top option, whether confidence rises or falls, and how Brier scores change. I’ll also verify that only the intended 384 norm gains differ from the original weights.

```python
# Hypothesis: selective final-branch reweighting changes rankings and improves probability error.
def readarr(k,name):
    return np.load(work/f"fwd_{k}"/f"{name}.npy")
audit=[]
for ko,ke in [(19,20),(21,22),(23,24)]:
    p=readarr(ko,"probabilities"); q=readarr(ke,"probabilities")
    m=readarr(ko,"option_mask")
    assert np.array_equal(m,readarr(ke,"option_mask"))
    flip=np.nanargmax(p,axis=1)!=np.nanargmax(q,axis=1)
    conf=np.nanmax(q,axis=1)-np.nanmax(p,axis=1)
    audit.append((flip,conf))
flips=np.concatenate([x[0] for x in audit]); conf=np.concatenate([x[1] for x in audit])
bo=np.array([float(re.search(r"Mean Brier ([0-9.]+)",r["feedback"])[1]) for r in r17o])
be=np.array([float(re.search(r"Mean Brier ([0-9.]+)",r["feedback"])[1]) for r in r17e])
print("Holdout top-option changes:",int(flips.sum()),"/",len(flips))
print("Confidence increased/decreased:",int((conf>0).sum()),int((conf<0).sum()))
print("Brier original/edited, paired change SE:",bo.mean(),be.mean(),stat(be-bo))
import torch
gn="model.language_model.layers.23.post_ffn_norm.weight"
delta=load(gn)-load_original(gn)
print("Changed entries:",int(torch.count_nonzero(delta)),"Finite:",bool(torch.isfinite(load(gn)).all()))
print("Largest absolute gain changes:",torch.topk(delta.abs(),12).indices.tolist())
print("Only edited tensor:",edited())
json.dump({"type":"decide","model":"edited","problems":[r["id"] for r in [json.loads(s) for s in (work/"gen_1.jsonl").read_text().splitlines()]]},
          (work/"final_practice.json").open("w"))
with (work/"notes.md").open("a") as f:
    f.write(f"Full450 holdout readout audit: {flips.sum()}/2250 top-option changes; confidence increases {(conf>0).sum()}, decreases {(conf<0).sum()}. Mean rounded-feedback Brier {bo.mean():.5f} -> {be.mean():.5f}; paired delta SE {np.std(be-bo,ddof=1)/np.sqrt(450):.5f}. Only384 entries in last post_ffn_norm changed, finite.\n"
            "Pass25 final DECIDE edited all900 practice examples for reporting only; locked validated candidate C-strength2 unchanged. No further selection.\n")
```
