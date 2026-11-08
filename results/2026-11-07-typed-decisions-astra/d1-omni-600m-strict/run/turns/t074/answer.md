I’ll compare the gated and ungated edits, focusing on whether unrelated logits return to the validated base repair while the severity gains remain.
```python
# Hypothesis: question gating causally suppresses off-target effects without losing useful severity corrections.
arr18,rowinfo18,meta18=load_read(18,read_ids_b)
read_summary(arr18,rowinfo18,arr10,rowinfo10)
print("Compared with original:")
read_summary(arr18,rowinfo18,arr3,rowinfo3)
target_idx=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"]
other_idx=[r["i"] for r in rowinfo3 if r["i"] not in target_idx]
for label,ar in [("ungated",arr17),("gated",arr18)]:
    for target,ix in [("severity",target_idx),("other",other_idx)]:
        diffs=[]
        for i in ix:
            n=len(rowinfo3[i]["opts"])
            delta=ar["option_logits"][i,:n]-arr10["option_logits"][i,:n]
            diffs.extend(delta-delta.mean())
        print(label,target,"centered logit-change RMS/max",round(float(np.sqrt(np.mean(np.square(diffs)))),5),
              round(float(np.max(np.abs(diffs))),5))
backup_edits("security_gated_moderate")
```
This comparison uses only discovery-B results; confirmation and validation records are not used to construct the gate.
