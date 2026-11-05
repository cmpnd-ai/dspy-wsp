The first case shows two useful confusions: the model prefers “observe” over “human review” and “success” over “partial.” I’ll check whether these recur, inspect other failing cases, and create a fixed fit/test split for head-level experiments.

```python
# Hypothesis: the scorer misranks semantically close options despite useful contextual representations.
stats = collections.defaultdict(list)
conf = collections.defaultdict(collections.Counter)
for r in records:
    ex = byid[r["id"]]
    ans = json.loads(r["calls"][0]["completion"])
    for q,g in ex["labels"]["gold"].items():
        p = ans[q].get("probabilities", {"false":1-ans[q].get("noul",0), "true":ans[q].get("noul",0)})
        pred = max(p,key=p.get); target=max(g["probabilities"],key=g["probabilities"].get)
        key=(ex["inputs"]["workflow"],q)
        stats[key].append(pred==target)
        if pred!=target: conf[key][(pred,target)]+=1
for k,v in stats.items():
    print(k, round(np.mean(v),3), conf[k].most_common(3))
rng=np.random.default_rng(42)
perm=rng.permutation(len(examples))
fit_ids=set(examples[i]["id"] for i in perm[:450])
test_ids=set(byid)-fit_ids
ordered_ids=[examples[i]["id"] for i in perm]
for i in range(5):
    req={"type":"read","model":"original","from":{"pass":1,"ids":ordered_ids[i*180:(i+1)*180]}}
    json.dump(req,open(work/f"read_batch_{i}.json","w"))
print("\nRepresentative failures:")
for wf in sorted({x["inputs"]["workflow"] for x in examples}):
    r=next(r for r in records if r["inputs"]["workflow"]==wf and r["score"]<=.4)
    print(r["id"],json.dumps(r["inputs"]["state"]),r["feedback"])
with notes.open("a") as f:
    f.write("\nBaseline: 0.56556 ± 0.00714 (900 cases). Fixed random seed 42 split: 450 fit, 450 independent practice test. Read passes planned in five batches of 180 (900 questions each); original, no edits.\n")
```
