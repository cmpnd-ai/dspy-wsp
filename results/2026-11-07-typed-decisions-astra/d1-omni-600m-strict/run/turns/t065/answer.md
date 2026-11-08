I’ll compare the higher dose with both the lower dose and the validated base repair. I’ll also inspect the remaining severity errors to see whether they reflect missing cues or an overstrong readout.
I’ll compare the higher dose with both the lower dose and the validated base repair. I’ll also inspect the remaining severity errors to see whether they reflect missing cues or an overstrong readout.
```python
# Hypothesis: the high-dose security edit improves cue-sensitive severity decisions, but its collateral errors may limit net gain.
arr16,rowinfo16,meta16=load_read(16,read_ids_b)
read_summary(arr16,rowinfo16,arr10,rowinfo10)
print("Versus original:")
read_summary(arr16,rowinfo16,arr3,rowinfo3)
print("Severity remaining errors on discovery B:")
for r in rowinfo16:
    if r["wf"]=="security_incidents" and r["q"]=="severity" and not r["ok"]:
        st=by_id[r["id"]]["inputs"]["state"]
        print(r["id"],"gold/pred",r["opts"][r["gi"]],r["opts"][r["pi"]],
              st["context"]["asset_criticality"],st["principal"]["privileges"],st["alert"]["rule"],
              "allowlisted",st["context"]["source_on_allowlist"])
for priv in [False,True]:
    ix=[r["i"] for r in rowinfo3 if r["wf"]=="security_incidents" and r["q"]=="severity"
        and ("deploy:production" in by_id[r["id"]]["inputs"]["state"]["principal"]["privileges"])==priv
        and by_id[r["id"]]["inputs"]["state"]["context"]["asset_criticality"]!="crown_jewel"]
    print("production",priv,"3-2 margins base/low/high",
          [round(float(np.mean(ar["option_logits"][ix,3]-ar["option_logits"][ix,2])),3)
           for ar in [arr10,arr15,arr16]])
backup_edits("security_high")
```
