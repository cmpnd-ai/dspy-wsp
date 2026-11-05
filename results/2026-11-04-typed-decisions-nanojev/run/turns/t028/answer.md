I’ll finish the fixed-offset comparison and inspect a few fitting states to design controlled cue changes. Those paired inputs can test whether the repair reacts to evidence such as constraint violations and delivery problems.

```python
# Hypothesis: controlled changes to state evidence should alter repaired logits in the expected direction.
testb1=read_batch(10,test_ids[200:])
ordered=[r["id"] for r in records if r["id"] in set(test_ids[200:])]
testb1["ids"]=[ordered[r["request"]] for r in testb1["rows"]]
testdesign1=design(testb1)
for b,d in [(testb1,testdesign1)]:
    for i,(h,z,p,r) in enumerate(d[2]):
        key=(byid[b["ids"][i]]["inputs"]["workflow"],r["question"])
        for label,delta in [("baseline",0),("fixed_offsets",mean_corrections[key]),("full_repair",h@dw600)]:
            scores[label].append(np.argmax(z+delta)==p.argmax())
for k,v in scores.items(): print(k,np.mean(v))
for wf in sorted(set(e["inputs"]["workflow"] for e in examples)):
    e=next(byid[i] for i in fit_ids if byid[i]["inputs"]["workflow"]==wf)
    print(wf,e["id"],json.dumps(e["inputs"]["state"],indent=2)[:3500])
with open(work/"notes.md","a") as f:
    f.write("\nMechanism check on independent 300 practice cases: fixed per-question-option average corrections versus full state-dependent correction. No new weight selection.\n")
```
