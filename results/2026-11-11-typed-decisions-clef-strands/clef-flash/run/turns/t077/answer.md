I’ll compare answer stability across the metadata variants, then verify the selected repair through the forward tool. This is a robustness check only; I won’t use these probes to fit another edit.


```python
# Hypothesis: irrelevant archive metadata should not erase the selected repair's gains or cause substantially more answer flips.
nuis_mm,nuis_ff,nuis_mk,nuis_l0=read_probe_arrays(20)
nuis_l2=simulate_probe_scoring(nuis_ff,nuis_mk,nuis_l0,repair_b2["tensors"])
nuis_lookup={(nuisance_meta[m["request"]]["seed"],nuisance_meta[m["request"]]["level"],m["question"]):i for i,m in enumerate(nuis_mm)}
nuis_summary={}
for level in ["original","short_metadata","long_metadata"]:
    scores=[[],[]]
    flips=[[],[]]
    shifts=[[],[]]
    for ii,m in enumerate(nuis_mm):
        pm=nuisance_meta[m["request"]]
        if pm["level"]!=level:
            continue
        gold=byid[pm["seed"]]["labels"]["gold"][m["question"]]["probabilities"]
        target=max(gold,key=gold.get)
        ref=nuis_lookup[(pm["seed"],"original",m["question"])]
        for j,ll in enumerate([nuis_l0,nuis_l2]):
            pp=int(ll[ii].argmax())
            scores[j].append(m["options"][pp]==target)
            flips[j].append(pp!=int(ll[ref].argmax()))
            shifts[j].append(float((ll[ii].softmax(-1)-ll[ref].softmax(-1)).abs().sum()/2))
    nuis_summary[level]={"accuracy_original_edited":[float(np.mean(x)) for x in scores],
                         "flip_rate_original_edited":[float(np.mean(x)) for x in flips],
                         "probability_TV_original_edited":[float(np.mean(x)) for x in shifts]}
print("Metadata robustness (40 seed records, descriptive only):")
for k,v in nuis_summary.items(): print(k,v)
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 20 metadata diagnostic\nOriginal backbone features collected for40 seeds x3 metadata conditions. Local best_2 simulation, descriptive only: "+str(nuis_summary)+". No fitting or candidate selection from this diagnostic.\n")
    f.write("\n## Planned pass 21\nExact edits best_2 unchanged: targeted_candidate.pt, 24 scorer neurons, three tensors. READ edited on identical120 nuisance requests, nuisance_probes_edited.json. Verify local predictions and unchanged scorer inputs.\n")
```
