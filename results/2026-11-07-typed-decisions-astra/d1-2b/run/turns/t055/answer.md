To recover the extra records’ final-write features without running the backbone locally, I’ll use the same complementary gain probes as before. First, I’ll request their original readouts and summarize where the confirmed 540-record repair helps on practice tests.
```python
# Hypothesis: the additional records contain useful state-dependent write features for the same option-contrast repair.
extra_read_request = {
    "type":"read", "model":"original",
    "from":{"pass":1,"ids":expand_ids},
    "layers":[23,24]
}
json.dump(extra_read_request, (work_dir/"read_expansion180.json").open("w"))
summary540 = collections.defaultdict(list)
for rec in records19:
    eid = rec["id"]
    golds = by_id[eid]["labels"]["gold"]
    oldp, newp = call_probs(base_by_id[eid]), call_probs(rec)
    for qid, gold in golds.items():
        winner = max(gold["probabilities"],key=gold["probabilities"].get)
        oldwin = max(oldp[qid],key=oldp[qid].get)
        newwin = max(newp[qid],key=newp[qid].get)
        summary540[(rec["inputs"]["workflow"],qid)].append((oldwin==winner,newwin==winner))
for key in sorted(summary540):
    arr = np.array(summary540[key],dtype=float)
    print("/".join(key), "original %.3f -> repair %.3f" % tuple(arr.mean(axis=0)))
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPlanned pass 21: READ original 180 additional records listed in final_expansion_split.json added_fit, using read_expansion180.json; 900 questions. Current installed edit is expanded540 W2 only, but model=original ignores it. No labels from remaining_check or validation enter the new fit.\n")
```