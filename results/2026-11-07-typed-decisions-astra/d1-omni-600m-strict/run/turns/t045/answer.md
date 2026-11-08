I’ll now compare the correctly aligned results at violation gains 0, 2 and 4. This will determine whether to retain the stronger readout or revert to gain 2.
```python
# Hypothesis: violation gain4 should only be retained if it improves the intended cue-sensitive questions without substantial collateral damage.
arr11, rowinfo11, meta11 = load_read(11, read_ids_b)
read_summary(arr11, rowinfo11, arr3, rowinfo3)
print("Increment over violation gain2:")
read_summary(arr11, rowinfo11, arr10, rowinfo10)
for flag in [False, True]:
    print("\nConstraint violations:", flag)
    for qid in ["action", "needs_review", "outcome", "risk", "urgency"]:
        inds = [r["i"] for r in rowinfo3 if r["wf"]=="agent_trace_observability" and r["q"]==qid
                and bool(by_id[r["id"]]["inputs"]["state"]["trace_summary"]["constraint_violations"]) == flag]
        print(qid, "n", len(inds), "accuracy [base,pay4,viol2,viol4]",
              [round(float(np.mean([int(ar["probabilities"][i,:len(rowinfo3[i]["opts"])].argmax()) == rowinfo3[i]["gi"] for i in inds])),3)
               for ar in [arr3, arr8, arr10, arr11]])
print("features unchanged:", np.array_equal(arr11["head_inputs"],arr3["head_inputs"]))
print("disc_rows sample:", str(disc_rows[0])[:850])
backup_edits("payment4_violation4")
note("\nPass 11 result: payment4+violation4 read completed; comparing discovery-B dose response against gains0/2. Checkpoint payment4_violation4.")
```
