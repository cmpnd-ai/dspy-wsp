I’ll assess the gain-4 dose-response and inspect the agent-trace errors next. Agent records have explicit violation, irreversible-action, and tool-error counts, so matched changes to those fields can test whether another important state cue is being underread.
The analysis below compares gain 4 with gain 2, then summarizes how agent-trace predictions respond to the structured counters on the discovery set.
```python
# Hypothesis: explicit operational counters are another underread state cue, analogous to payment deadlines.
arr8,rowinfo8,meta8=load_read(8,read_ids_b)
read_summary(arr8,rowinfo8,arr3,rowinfo3)
deadline_summary(arr8,rowinfo8)
backup_edits("payment_gain4")
print("\nAgent state counter slices (discovery only):")
agent_ids=[eid for eid in discovery_ids if by_id[eid]["inputs"]["workflow"]=="agent_trace_observability"]
for field in ["constraint_violations","irreversible_actions","tool_errors"]:
    print("\n",field)
    for flag in [False,True]:
        ids=[eid for eid in agent_ids if (by_id[eid]["inputs"]["state"]["trace_summary"][field]>0)==flag]
        rr=[r for r in disc_rows if r["id"] in ids]
        print("positive",flag,"n",len(ids))
        for q in ["action","outcome","risk","urgency"]:
            rows=[r for r in rr if r["q"]==q]
            print(q, "gold",dict(collections.Counter(r["gold"] for r in rows)),
                  "pred",dict(collections.Counter(r["pred"] for r in rows)))
```
