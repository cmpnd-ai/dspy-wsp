Twice strength gives a 6.74-point fit gain, but also more regressions. Before selecting an edit, I’ll expand the causal measurements to 200 additional fit records. This may produce a more stable correction than simply increasing the strength fitted on the first 200.

```python
# Hypothesis: more fit-only causal measurements reduce variance in the sparse gain correction.
seen={e["id"] for e in aligned}
readfit2=[e for e in fit if e["id"] not in seen][:200]
drop_edit(gn)
req2={"type":"read","model":"original","from":{"pass":1,"ids":[e["id"] for e in readfit2]}}
json.dump(req2,open(work/"read_more.json","w"))
note("Pass7 strength2 fit gain +.067411 SE .009955. Next pass8 original READ on 200 additional fit-only cases, no edits. Holdout untouched.")
print("additional workflows",collections.Counter(e["inputs"]["workflow"] for e in readfit2))
```
