I’ll measure the refined edit’s overall gain and separate changes on low-, high- and critical-impact security cases. This will show whether the new low-impact readout helps or mostly adds noise.
```python
# Hypothesis: narrative-protected security readouts improve severity selectively and reduce earlier high-dose collateral errors.
arr17,rowinfo17,meta17=load_read(17,read_ids_b)
read_summary(arr17,rowinfo17,arr10,rowinfo10)
print("Versus original:")
read_summary(arr17,rowinfo17,arr3,rowinfo3)
print("Security severity groups:")
for crit in ["low","medium","high","crown_jewel"]:
    for priv in [False,True]:
        ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
            and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]==crit
            and ("deploy:production" in by_id[r["id"]]["inputs"]["state"]["principal"]["privileges"])==priv]
        if ix:
            print(crit,priv,"n",len(ix),"accuracy base/oldlow/refined",
                  [round(float(np.mean([int(ar["option_logits"][i,:5].argmax())==rowinfo3[i]["gi"] for i in ix])),3)
                   for ar in [arr10,arr15,arr17]],
                  "pred",collections.Counter(int(arr17["option_logits"][i,:5].argmax()) for i in ix))
backup_edits("security_specific_moderate")
```
