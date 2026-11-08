Strength three nearly eliminates the gain, confirming a bounded response. I’ll select candidate C at strength two based on fit results and test it on 450 previously unused practice records. No further fitting will use those records.

```python
# Hypothesis: the selected strength-two correction generalizes to records not used for fitting or strength selection.
save_edit(gn,g0*(1+torch.tensor(2*v3,dtype=torch.float32)))
holdconfirm=holdids[:450]
json.dump({"type":"decide","model":"both","problems":holdconfirm},open(work/"hold_decide.json","w"))
note("Pass16 C strength3 fit +.00625 SE .014045, saturation confirmed. LOCK selected candidate C strength2 (ridge.1,384dims,400 fit cases). Pass17 independent practice holdout450 of452; no further fitting or strength selection on holdout.")
```
