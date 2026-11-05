In the first trace, the model recommends stopping despite zero recorded violations or irreversible actions. I’ll quantify whether this overreaction recurs and collect EOS representations from 200 fitting cases to test whether the head misreads evidence already present in the backbone.

```python
# Hypothesis: excessive intervention and severity reflect a misaligned scalar readout, not missing state information.
stats=collections.defaultdict(list)
confusions=collections.defaultdict(collections.Counter)
for r in records:
    gold=byid[r["id"]]["labels"]["gold"]
    completion=json.loads(r["calls"][0]["completion"])
    for q,g in gold.items():
        out=completion[q]
        p=out.get("probabilities",{"false":1-out.get("noul",0),"true":out.get("noul",0)})
        pred=max(p,key=p.get); target=max(g["probabilities"],key=g["probabilities"].get)
        key=(r["inputs"]["workflow"],q)
        stats[key].append(pred==target)
        confusions[key][(target,pred)]+=1
for k,v in stats.items():
    print(k,round(np.mean(v),3),"top errors",[(a,n) for a,n in confusions[k].most_common() if a[0]!=a[1]][:3])
fit_chunks=[fit_ids[i:i+200] for i in range(0,600,200)]
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":fit_chunks[0]}},open(work/"read_fit0.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nBaseline: 0.36422 ± 0.00764 item SE. First trace wrongly stops despite zero violations/irreversible actions.\nPass 2 planned: original read, first 200 fitting IDs, no edits, inspect EOS/head representations.\n")
```
