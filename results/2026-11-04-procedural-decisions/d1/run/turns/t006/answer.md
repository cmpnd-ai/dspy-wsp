I’ll tabulate the question-level errors and print a few failures from each workflow. I’ll also set aside a fixed practice split so the eventual repair is tested on examples not used to derive it.

```python
# Hypothesis: the scorer overweights semantic severity/urgency cues and underweights explicit state and rule evidence.
byid={x["id"]:x for x in data}
stats=collections.defaultdict(list)
conf=collections.defaultdict(collections.Counter)
for r in base:
    c=json.loads(r["calls"][0]["completion"])
    for q,g in byid[r["id"]]["labels"]["gold"].items():
        p=c[q].get("probabilities")
        if p is None: p={"false":1-c[q]["noul"],"true":c[q]["noul"]}
        pred=max(p,key=p.get); target=max(g["probabilities"],key=g["probabilities"].get)
        stats[q].append(pred==target); conf[q][(target,pred)]+=1
for q,v in stats.items(): print(q, len(v),round(np.mean(v),3),dict(conf[q]))
for wf in sorted(set(x["inputs"]["workflow"] for x in data)):
    print("\nWORKFLOW",wf)
    for r in [r for r in base if r["inputs"]["workflow"]==wf and r["score"]<1][:2]:
        print(r["id"],json.dumps(r["inputs"]["state"]),r["feedback"])
rng=np.random.default_rng(42)
fitids=[]; testids=[]
for wf in sorted(set(x["inputs"]["workflow"] for x in data)):
    ids=[x["id"] for x in data if x["inputs"]["workflow"]==wf]
    rng.shuffle(ids); fitids+=ids[:140]; testids+=ids[140:280]
with open(work/"notes.md","a") as f:
    f.write("\nBaseline 900: accuracy .51593 SE .01233. Fixed split: 140 per workflow fit, 140 per workflow untouched practice test, 20 per workflow reserve. Next read original head features for 420 fit cases (980 questions). No edits.\n")
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":fitids}},open(work/"read_fit.json","w"))
```
