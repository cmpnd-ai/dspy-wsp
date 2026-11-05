In the first failed record, the model selects the initial severity (“critical”) rather than the updated severity (“degraded”), and predicts open despite two resolve events. I’ll quantify these patterns and collect decision-head inputs from a fixed fitting subset to test whether the final readout misses information already present in the backbone.

```python
# Hypothesis: a targeted correction to the scalar head can recover state/rule evidence currently outweighed by answer priors.
stats = collections.defaultdict(list)
confusions = collections.defaultdict(collections.Counter)
for r in records:
    ans = json.loads(r["calls"][0]["completion"])
    for q,g in byid[r["id"]]["labels"]["gold"].items():
        a=ans[q]
        p=a.get("probabilities", {"false":1-a.get("noul",0),"true":a.get("noul",0)})
        pred=max(p,key=p.get); target=max(g["probabilities"],key=g["probabilities"].get)
        stats[q].append(pred==target); confusions[q][(target,pred)]+=1
for q in stats:
    print(q, "accuracy",round(np.mean(stats[q]),3), "confusion",dict(confusions[q]))
for wf in ("multi_view_adjudication","partial_observation_calibration"):
    r=next(r for r in records if r["inputs"]["workflow"]==wf and r["score"]<0.5)
    print("\nFAILURE",r["id"],json.dumps(r["inputs"]["state"]),r["feedback"])
rng=random.Random(731)
fit_ids=[]; test_ids=[]
for wf in sorted(set(x["inputs"]["workflow"] for x in data)):
    ids=[x["id"] for x in data if x["inputs"]["workflow"]==wf]
    rng.shuffle(ids); fit_ids+=ids[:140]; test_ids+=ids[140:]
with open(notes,"a") as f:
    f.write("\nBaseline all 900: accuracy .4800, SE .01263. Per-question: "+str({q:float(np.mean(v)) for q,v in stats.items()})+
            "\nNo edits. Read pass 2: fixed fitting subset, 140 examples per workflow (420 total, 980 questions), seed 731. Test subset remaining 480 excluded from fitting.\n")
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":fit_ids}},open(work/"read_fit.json","w"))
```
