I’ll identify the largest confusion pairs and compare their states. I’ll also reserve 300 practice records for a paired confirmation, separate from records used to fit or select an edit.

```python
# Hypothesis: particular option margins are systematically wrong even when the state contains decisive evidence.
basebyid = {str(r["id"]): r for r in base}
rows = []
for r in base:
    ex = byid[str(r["id"])]
    ans = json.loads(r["calls"][0]["completion"])
    for qid,gold in ex["labels"]["gold"].items():
        p = ans[qid]["probabilities"]
        target = max(gold["probabilities"], key=gold["probabilities"].get)
        best = max(p, key=p.get)
        rows.append(dict(id=str(r["id"]), wf=r["inputs"]["workflow"], q=qid, target=target,
                         pred=best, conf=p[best], correct=(target==best), probs=p))
groups = collections.defaultdict(list)
for r in rows:
    groups[r["wf"],r["q"]].append(r)
for (wf,q), rr in sorted(groups.items()):
    errors = [r for r in rr if not r["correct"]]
    confusions = collections.Counter((r["target"],r["pred"]) for r in errors)
    balance = collections.Counter(r["target"] for r in rr)
    print(f"{wf[:9]:9s} {q:23s} acc={np.mean([r['correct'] for r in rr]):.3f} err_conf={np.mean([r['conf'] for r in errors]):.2f} gold={dict(balance)} confused={confusions.most_common(4)}")
print("\nHead forward source:\n", "\n".join(source.splitlines()[242:462]))

rng = random.Random(271828)
splits = {"fit":[], "tune":[], "test":[]}
for wf in sorted({d["inputs"]["workflow"] for d in data}):
    ids = [str(d["id"]) for d in data if d["inputs"]["workflow"]==wf]
    rng.shuffle(ids)
    splits["fit"] += ids[:125]
    splits["tune"] += ids[125:150]
    splits["test"] += ids[150:]
for ids in splits.values():
    rng.shuffle(ids)
json.dump(splits, (WORK/"practice_splits.json").open("w"))
note("## Practice split (fixed seed 271828)\nFit 500 (125/workflow), tune 100 (25/workflow), untouched paired test 300 (75/workflow). IDs in practice_splits.json. Baseline diagnosis summarizes all records; no edit fitting or parameter selection will use test labels.")
print("Split sizes:", {k:len(v) for k,v in splits.items()})
```

I’ll identify the largest confusion pairs and inspect how the head constructs their logits. I’ll also reserve 300 practice records for confirmation, separate from those used to fit or select an edit.

```python
# Hypothesis: particular option margins are systematically wrong even when the state contains decisive evidence.
basebyid = {str(r["id"]): r for r in base}
rows = []
for r in base:
    ex = byid[str(r["id"])]
    ans = json.loads(r["calls"][0]["completion"])
    for qid,gold in ex["labels"]["gold"].items():
        p = ans[qid]["probabilities"]
        target = max(gold["probabilities"], key=gold["probabilities"].get)
        best = max(p, key=p.get)
        rows.append(dict(id=str(r["id"]), wf=r["inputs"]["workflow"], q=qid, target=target,
                         pred=best, conf=p[best], correct=(target==best), probs=p))
groups = collections.defaultdict(list)
for r in rows:
    groups[r["wf"],r["q"]].append(r)
for (wf,q), rr in sorted(groups.items()):
    errors = [r for r in rr if not r["correct"]]
    confusions = collections.Counter((r["target"],r["pred"]) for r in errors)
    balance = collections.Counter(r["target"] for r in rr)
    print(f"{wf[:9]:9s} {q:23s} acc={np.mean([r['correct'] for r in rr]):.3f} err_conf={np.mean([r['conf'] for r in errors]):.2f} gold={dict(balance)} confused={confusions.most_common(4)}")
print("\nHead forward source:\n", "\n".join(source.splitlines()[242:462]))

rng = random.Random(271828)
splits = {"fit":[], "tune":[], "test":[]}
for wf in sorted({d["inputs"]["workflow"] for d in data}):
    ids = [str(d["id"]) for d in data if d["inputs"]["workflow"]==wf]
    rng.shuffle(ids)
    splits["fit"] += ids[:125]
    splits["tune"] += ids[125:150]
    splits["test"] += ids[150:]
for ids in splits.values():
    rng.shuffle(ids)
json.dump(splits, (WORK/"practice_splits.json").open("w"))
note("## Practice split (fixed seed 271828)\nFit 500 (125/workflow), tune 100 (25/workflow), paired test 300 (75/workflow). IDs in practice_splits.json. Baseline diagnosis summarizes all records; no edit fitting or parameter selection will use test labels.")
print("Split sizes:", {k:len(v) for k,v in splits.items()})
```
