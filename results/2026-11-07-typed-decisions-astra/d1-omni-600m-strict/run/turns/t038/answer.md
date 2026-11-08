I’ll measure the violation readout’s incremental effect, especially on harmful outcomes. I’ll also check whether non-violated traces remain stable; that distinction is essential to the proposed mechanism.
The analysis below compares the combined edit with both the original model and the payment-only edit, then splits agent results by whether a violation occurred.
```python
# Hypothesis: the new violation cue improves violated traces while leaving non-violated traces largely unchanged.
arr10,rowinfo10,meta10=load_read(10,read_ids_b)
read_summary(arr10,rowinfo10,arr3,rowinfo3)
agent_b=[r for r in rowinfo10 if r["wf"]=="agent_trace_observability"]
for flag in [False,True]:
    print("Violations present",flag)
    for q in ["action","needs_review","outcome","risk","urgency"]:
        rs=[r for r in agent_b if r["q"]==q and
            (by_id[r["id"]]["inputs"]["state"]["trace_summary"]["constraint_violations"]>0)==flag]
        ii=[r["i"] for r in rs]
        print(q,"n",len(rs),"orig/pay/combined",
              np.round([np.mean([inf[i]["ok"] for i in ii]) for inf in [rowinfo3,rowinfo8,rowinfo10]],3),
              "new pred",dict(collections.Counter(r["opts"][r["pi"]] for r in rs)))
backup_edits("payment4_violation2")
```
