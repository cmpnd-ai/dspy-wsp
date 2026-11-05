I’ll summarize the measured dose-response and inspect how the remaining ablations are installed. These tests characterize the repair; they will not replace the validated candidate.

```python
# Hypothesis: accuracy and controlled-cue logits change systematically along the repaired readout direction.
print("install signature", inspect.signature(install_job))
print("example", str(byid[panel_ids[0]])[:750])
print("gold", byid[panel_ids[0]]["labels"]["gold"])
for pn in [13,14,15,16,17]:
    pm=json.load((work/f"fwd_{pn}"/"meta.json").open())["rows"]
    probs=np.load(work/f"fwd_{pn}"/"probabilities.npy")
    hits=[]
    for ri,row in enumerate(pm[:200]):
        gg=byid[panel_ids[row["request"]]]["labels"]["gold"][row["question"]]["probabilities"]
        hits.append(row["options"][np.nanargmax(probs[ri])]==bi.max(gg,key=gg.get))
    print(pn, "panel accuracy", np.mean(hits))
print("synthetic full responses")
for ri,row in enumerate(pm[200:],200):
    if row["question"] in ["action","category","risk"]:
        print(synlabels[row["request"]-40],row["question"],dict(zip(row["options"],np.round(probs[ri],3))))
```