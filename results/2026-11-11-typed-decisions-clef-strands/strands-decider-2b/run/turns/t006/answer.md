Several failures are directional: invoices with delivery exceptions are treated as matching, agent runs without completion evidence are called successful, and invoice urgency is shifted upward. I’ll read the pointer-head inputs to test whether the backbone represents these distinctions but the head reads them poorly.

```python
# Hypothesis: state evidence survives in the final representations but is underweighted by the shared query/key readout.
wf_list = sorted({ex["inputs"]["workflow"] for ex in examples})
fit_batches = []
for batch in range(3):
    ids = []
    for wf in wf_list:
        wf_ids = [i for i in splits["fit"] if byid[i]["inputs"]["workflow"] == wf]
        ids += wf_ids[batch*45:(batch+1)*45]
    fit_batches.append(ids)
read_plan = [("fit_a", fit_batches[0]), ("fit_b", fit_batches[1]), ("fit_c", fit_batches[2]),
             ("dev", splits["dev"]), ("test", splits["test"])]
for label, ids in read_plan:
    req = {"type": "read", "model": "original", "from": {"pass": 1, "ids": ids}}
    json.dump(req, open(WORK / f"read_{label}.json", "w"))
lognote("## Before pass 2\nExact edit: none. READ original fit_a, 180 fit items (45/workflow), 900 questions; "
        "IDs and request in read_fit_a.json. Hypothesis: final states contain workflow-specific evidence that "
        "the shared pointer readout underweights. Planned reads fit_b, fit_c, dev, test are disjoint; "
        "test representations will be held unused until an edit is fixed.")
print("Prepared read requests:", [(label,len(ids)) for label,ids in read_plan])
for wf,qid in [("customer_service","action"),("invoice_processing","urgency"),("security_incidents","true_positive")]:
    rr = [r for r in groups[(wf,qid)] if not r["ok"] and id_split[r["id"]]=="fit"]
    r = sorted(rr,key=lambda z:z["confidence"],reverse=True)[0]
    print("\nRepresentative failure:",r["id"],wf,qid)
    print("Probabilities",r["p"],"Gold",r["gold"])
    print("Actual request state:",json.dumps(json.loads(recbyid[r["id"]]["calls"][0]["prompt"])["state"],ensure_ascii=False)[:1900])
```
