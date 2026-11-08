I’ll check whether the low-dose edit selectively improves security severity and whether it leaves the earlier payment and violation gains intact.
```python
# Hypothesis: security cue readouts improve high/critical severity decisions without shifting unrelated question rankings.
arr15,rowinfo15,meta15=load_read(15,read_ids_b)
read_summary(arr15,rowinfo15,arr10,rowinfo10)
print("Overall compared with original:")
read_summary(arr15,rowinfo15,arr3,rowinfo3)
for crit in ["low","medium","high","crown_jewel"]:
    ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
        and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]==crit]
    print(crit,"n",len(ix),"accuracy original/base/low",
          [round(float(np.mean([int(ar["option_logits"][i,:5].argmax())==rowinfo3[i]["gi"] for i in ix])),3)
           for ar in [arr3,arr10,arr15]],
          "edited predictions",collections.Counter(int(arr15["option_logits"][i,:5].argmax()) for i in ix))
for priv in [False,True]:
    ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
        and ("deploy:production" in by_id[r["id"]]["inputs"]["state"]["principal"]["privileges"])==priv
        and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]!="crown_jewel"]
    print("production access",priv,"n",len(ix),"mean 3-2 margins base/low",
          [round(float(np.mean(ar["option_logits"][ix,3]-ar["option_logits"][ix,2])),3) for ar in [arr10,arr15]])
backup_edits("security_low")
```
