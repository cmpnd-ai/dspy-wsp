The 1.5-strength correction overshoots, as predicted. I’ll restore full strength and examine the matched-cue reads and question-level results to distinguish contextual repair from a change in answer priors.

```python
# Hypothesis: the correction changes state-dependent option comparisons rather than only answer priors.
save_edit("head.scorer.3.weight", load_original("head.scorer.3.weight") + torch.tensor(delta_all,dtype=torch.float32).reshape(1,-1))
print("cue arrays", [p.name for p in (work/"fwd_11").iterdir()])
cm=json.load(open(work/"fwd_11"/"meta.json"))
print("metadata keys",cm.keys())
print(str({k:v for k,v in cm.items() if k!="rows"})[:5000])
print("cue request sample", str(cue_requests)[:2500])
print("record sample",str(records[0])[:2400])
print("row sample",str(rows[0])[:1500])
with (work/"notes.md").open("a") as f:
    f.write("\nPass19 1.5-dose +.05022±.01304 vs full +.09289 on same450; overshoot confirmed. Restored full all900 delta for diagnostic reads. Next inspect recorded cue responses and workflow gains.\n")
```
