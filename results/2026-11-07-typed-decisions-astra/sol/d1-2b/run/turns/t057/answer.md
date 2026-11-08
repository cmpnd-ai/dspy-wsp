The Brier parser included the sentence-ending period. I’ll fix that parsing error and complete the audit; no weights have changed.

```python
# Hypothesis: corrected feedback parsing confirms probability-quality gains without changing weights.
bo=np.array([float(re.search(r"Mean Brier (\d+\.\d+)",r["feedback"])[1]) for r in r17o])
be=np.array([float(re.search(r"Mean Brier (\d+\.\d+)",r["feedback"])[1]) for r in r17e])
print("Holdout top-option changes:",int(flips.sum()),"/",len(flips))
print("Confidence increased/decreased:",int((conf>0).sum()),int((conf<0).sum()))
print("Brier original/edited, paired change SE:",bo.mean(),be.mean(),stat(be-bo))
import torch
gn="model.language_model.layers.23.post_ffn_norm.weight"
delta=load(gn)-load_original(gn)
print("Changed entries:",int(torch.count_nonzero(delta)),"Finite:",bool(torch.isfinite(load(gn)).all()))
print("Largest absolute gain changes:",torch.topk(delta.abs(),12).indices.tolist())
print("Only edited tensor:",edited())
json.dump({"type":"decide","model":"edited","problems":[json.loads(s)["id"] for s in (work/"gen_1.jsonl").read_text().splitlines()]},
          (work/"final_practice.json").open("w"))
with (work/"notes.md").open("a") as f:
    f.write(f"Full450 holdout readout audit: {flips.sum()}/2250 top-option changes; confidence increases {(conf>0).sum()}, decreases {(conf<0).sum()}. Mean rounded-feedback Brier {bo.mean():.5f} -> {be.mean():.5f}; paired delta SE {np.std(be-bo,ddof=1)/np.sqrt(450):.5f}. Only384 entries in last post_ffn_norm changed, finite.\n"
            "Pass25 final DECIDE edited all900 practice examples for reporting only; locked validated candidate C-strength2 unchanged. No further selection.\n")
```
