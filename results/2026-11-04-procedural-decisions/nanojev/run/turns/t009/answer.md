The initial fit generalizes poorly, so I won’t apply it. Before interpreting that as missing backbone information, I’ll verify request ordering: the read tool may preserve baseline order rather than the order of the supplied IDs.

```python
# Hypothesis: the read tool preserves baseline record order, so labels must follow that order too.
fit_order=[r["id"] for r in records if r["id"] in set(fit_ids)]
print("Requested first:",fit_ids[:5],"baseline-order first:",fit_order[:5])
for rid in [fit_ids[0],fit_order[0]]:
    a=json.loads(next(r for r in records if r["id"]==rid)["calls"][0]["completion"])
    print(rid,a["current_owner"]["probabilities"])
print("read first:",arr["probabilities"][0])
```
