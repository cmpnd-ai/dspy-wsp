The remaining errors include intent and open/closed status. I’ll summarize the workflow gains and inspect the fitted direction to choose a targeted ablation.
```python
# Hypothesis: gains come from state-dependent readout features, not merely option-specific offsets.
import inspect
for fun in (summarize_file, eval_all, scores_direction):
    try: print(inspect.getsource(fun))
    except Exception: print(fun)
for name in ["delta900","delta_prior","delta_evidence","delta_high","X","XX","YY","allrows","groups","design_groups"]:
    val=globals()[name]
    print(name, type(val), getattr(val,"shape",None), "length",len(val))
for k in (16,17,18):
    aa=[json.loads(x) for x in (p/f"gen_{k}_original.jsonl").read_text().splitlines()]
    bb=[json.loads(x) for x in (p/f"gen_{k}_edited.jsonl").read_text().splitlines()]
    da=np.array([b["score"]-a["score"] for a,b in zip(aa,bb)])
    print(k, np.mean([a["score"] for a in aa]), np.mean([b["score"] for b in bb]), "gain", da.mean(), "SE", da.std(ddof=1)/len(da)**.5)
print("norms",[(n,float(np.linalg.norm(globals()[n]))) for n in ["delta900","delta_prior","delta_evidence","delta_high"]])
print("Sample group", str(groups[0])[:1500])
```
