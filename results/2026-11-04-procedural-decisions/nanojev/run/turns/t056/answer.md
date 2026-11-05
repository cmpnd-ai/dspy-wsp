Full strength performs best in this sweep: 80.67%, versus 63.70% at half strength and 78.07% at 1.5×. I’ll record the comparison and use the last pass to reconfirm the submitted weights on held-out validation.
```python
# Hypothesis: the selected full readout repair retains its previously measured held-out gain.
dose_records={}
for k in (22,23,24):
    dose_records[k]=[json.loads(x) for x in (p/f"gen_{k}_edited.jsonl").read_text().splitlines()]
full={x["id"]:x["score"] for x in dose_records[24]}
for k in (22,23):
    ds=np.array([full[x["id"]]-x["score"] for x in dose_records[k]])
    print("Full minus",k,ds.mean(), "paired SE",ds.std(ddof=1)/np.sqrt(len(ds)))
with notes.open("a") as f:
    f.write("\nPass24 full on450 .47185→.80667 gain .33481±.02071; exceeds half (.63704) and1.5x(.78074). Current submission scalar.weight only, original+delta900 ridge.1. Pass25 final hidden validation on this exact restored edit; no other tensors modified. Full-data practice gains are in-sample; independent earlier 450 test +.19704±.02296 for initial420-fit edit; hidden validation is generalization evidence for final edit.\n")
print("Final edited tensors",edited())
```
