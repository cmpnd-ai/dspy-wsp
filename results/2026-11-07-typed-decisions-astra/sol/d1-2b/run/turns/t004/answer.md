The first record shows risk overstated despite zero violations and zero irreversible actions. I’ll check how common that pattern is, inspect other major confusions, and prepare a read pass to locate the relevant signal in the last MLP.

```python
# Hypothesis: final-MLP features contain cues that are misread by the option-token directions.
import re
exbyid={e["id"]:e for e in data}
stats=collections.defaultdict(list); conf=collections.defaultdict(collections.Counter)
for r in recs:
    wf=r["inputs"]["workflow"]
    for q,p,t in re.findall(r"(\w+): (?:right|WRONG) \(predicted (\S+) [\d.]+, gold (\S+) [\d.]+\)",r["feedback"]):
        stats[(wf,q)].append(p==t);conf[(wf,q)][(t,p)]+=1
for key,v in sorted(stats.items()):
    print(key, round(np.mean(v),3), "errors:",[(a,b,n) for (a,b),n in conf[key].most_common() if a!=b][:5])
for wf,q in [('agent_trace_observability','risk'),('invoice_processing','matches_order'),('security_incidents','credential_compromise')]:
    rr=[r for r in recs if r['id'] in fitids and r["inputs"]["workflow"]==wf and f"{q}: WRONG" in r["feedback"]][:2]
    for r in rr: print("\n",r["id"],json.dumps(r["inputs"]["state"]),r["feedback"])
readfit=fit[:50]+fit[112:162]+fit[224:274]+fit[336:386]
print("readfit",len(readfit),collections.Counter(e["inputs"]["workflow"] for e in readfit))
note("Baseline 900 practice accuracy .61333 SE .00825. Inspecting question confusions. Next READ original: 200 fit cases (1000 questions), last layer inputs; no edits.")
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":[e["id"] for e in readfit]},"layers":[23,24]},open(work/"read_request.json","w"))
```
