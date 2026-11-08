I’ll measure the gain-4 edit by workflow and split agent traces into those with and without constraint violations. The intended effect is stronger harmful/high-risk decisions only when violations are present.
```python
# Hypothesis: a useful violation direction should improve violated traces more than clean traces, with a measurable dose response.
meta11, arr11, rowinfo11 = load_read(11, read_ids_b)
read_summary(arr11, rowinfo11, arr3, rowinfo3)
print("Increment over violation gain2:")
read_summary(arr11, rowinfo11, arr10, rowinfo10)
for flag in [False, True]:
    print("\nConstraint violations:", flag)
    for qid in ["action", "needs_review", "outcome", "risk", "urgency"]:
        inds = [r["i"] for r in rowinfo3 if r["wf"]=="agent_trace_observability" and r["q"]==qid
                and bool(by_id[r["id"]]["inputs"]["state"]["trace_summary"]["constraint_violations"]) == flag]
        if not inds:
            continue
        print(qid, "n", len(inds), "accuracy [base,pay4,viol2,viol4]",
              [round(float(np.mean([int(ar["probabilities"][i,:len(rowinfo3[i]["opts"])].argmax()) == rowinfo3[i]["gi"] for i in inds])),3)
               for ar in [arr3, arr8, arr10, arr11]])
print("feature equality:", np.array_equal(arr11["head_inputs"],arr3["head_inputs"]))
print("discovery workflow accuracy/errors:")
for wf in sorted({r["wf"] for r in disc_rows}):
    rr = [r for r in disc_rows if r["wf"]==wf]
    print(wf, type(rr[0]), str(rr[0])[:300])
```
