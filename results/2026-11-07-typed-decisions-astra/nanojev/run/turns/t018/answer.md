Full strength improves development accuracy further, to 58.5%, confirming the expected response. I’ll freeze this candidate and test it on the 300 reserved practice records, which were not used to derive or select it.
```python
# Hypothesis: a genuine readout repair will retain its gain on records excluded from fitting and strength selection.
note("Forward6 full-strength soft direction dev150: original0.373333, edited0.585333, gain0.212000 SE0.025329, 96 improved/24 worsened. Dose response actual: strength0 ->37.33%, .5 ->52.00%, 1 ->58.53%.")
(WORK/"test_decide_request.json").write_text(json.dumps({"type":"decide","model":"both","problems":splits["test"]}))
note("Planned forward7: paired DECIDE reserved test300 IDs (test_decide_request.json), never used to fit or choose strength. Exact current edit scalar.weight=w_original+d_soft, saved candidate_weights/soft_full_scalar.pt; all other tensors original. Candidate frozen before this independent measurement.")
print("Test items",len(splits["test"]),"edits",edited())
```
