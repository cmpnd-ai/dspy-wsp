I’ll check the strong gated edit’s accuracy, severity margins and collateral effects. If it improves on the moderate dose, I’ll freeze it for an independent practice test rather than continue tuning on discovery B.
I’ll compare the strong gated edit with the base repair and the moderate dose, including probability error as a secondary check.
```python
# Hypothesis: stronger gated cue readouts improve severity with no unrelated ranking changes.
arr19,rowinfo19,meta19=load_read(19,read_ids_b)
read_summary(arr19,rowinfo19,arr10,rowinfo10)
print("Versus original:")
read_summary(arr19,rowinfo19,arr3,rowinfo3)
for crit in ["low","medium","high","crown_jewel"]:
    ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
        and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]==crit]
    print(crit,"n",len(ix),"accuracy base/moderate/strong",
          [round(float(np.mean([int(ar["option_logits"][i,:5].argmax())==rowinfo3[i]["gi"] for i in ix])),3)
           for ar in [arr10,arr18,arr19]])
for label,ar in [("original",arr3),("base",arr10),("gated moderate",arr18),("gated strong",arr19)]:
    briers=[np.sum((ar["probabilities"][r["i"],:len(r["opts"])]-r["g"])**2) for r in rowinfo3]
    print(label,"mean squared probability error",round(float(np.mean(briers)),5),
          "security severity",round(float(np.mean([briers[i] for i in target_idx])),5))
other_changed=[r["id"]+":"+r["q"] for r,s in zip(rowinfo19,rowinfo10) if r["i"] in other_idx and r["pi"]!=s["pi"]]
print("Unrelated ranking changes:",other_changed[:10],len(other_changed))
backup_edits("security_gated_strong")
```
