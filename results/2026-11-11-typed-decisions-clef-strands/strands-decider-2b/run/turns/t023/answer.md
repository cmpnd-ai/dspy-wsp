As predicted, 1.5× strength is worse than full strength. I’ll restore the full-strength candidate and test it on the 180 reserved practice cases, which were not used to derive the direction or choose its strength.

```python
# Hypothesis: the fixed full-strength key correction will retain a substantial gain on untouched practice cases.
lognote("Pass 8 actual paired dev, strength 1.5: .577778 -> .710000, gain .132222, "
        "SE .024723. Dose response strengths 0/.5/1/1.5 gives accuracy .5778/.7233/.7478/.7100; "
        "overshoot confirmed. Select strength 1.0 based only on development.")
save_rank1(1.0)
json.dump({"type":"decide","model":"both","problems":splits["test"]},open(WORK/"paired_test.json","w"))
lognote("## Before pass 9\nExact edit: k.weight = K0 + outer(u,v), lambda .001, rank one; no other edits. "
        "Paired DECIDE on reserved 180 practice-test IDs (45/workflow) in paired_test.json. "
        "These examples were not used to derive or select the edit; no test read has been performed.")
```
