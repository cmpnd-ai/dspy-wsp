I’ll collect the last 100 practice records, then assess the frozen readout recipe with cross-fitted predictions. That provides a practice estimate where each record is scored by a correction derived without that record.
I’ll request the last 100 records without changing the validated weights.
```python
# Hypothesis: more examples improve estimation of weak state-evidence directions without introducing new model components.
note("Forward15 read200 completed. Weighted contrasts did not materially beat simpler mixed target (.601333 vs .600000 dev). Keep simple recipe; no weighted or set-output edits saved.")
read16_ids=splits["test"][200:]
(WORK/"read16_request.json").write_text(json.dumps({"type":"read","model":"original","from":{"pass":1,"ids":read16_ids}}))
note("Planned forward16: original READ final100 test IDs, exact read16_request.json,500 questions. Current edit best2 scalar.weight=w_original+.75*d_mix600; no other tensors. Purpose: complete900 features for cross-fit measurement and fixed-recipe refit. Full900 candidate will require fresh hidden-validation confirmation.")
print("Remaining records",len(read16_ids),"current edits",edited())
```
