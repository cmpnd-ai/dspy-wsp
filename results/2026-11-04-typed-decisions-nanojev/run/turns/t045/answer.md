I’ll inspect the prepared diagnostic jobs, then complete the full-strength endpoint of the dose-response test.

```python
# Hypothesis: full-strength correction completes a causal dose-response on a fixed panel.
print("Jobs:", [(j[0], tuple(j[1].shape)) for j in jobs])
print("dose request:", (work/"dose_request.json").read_text()[:400])
print("panel ids:", panel_ids[:5], "count", len(panel_ids))
print("syn labels:", synlabels)
print("direction eigenvalues:", ea[:5], ea[-5:])
with (work/"notes.md").open("a") as f:
    f.write("\nRestored best_1 scalar after restart. Pass 17: full validated scalar on same 40 fitting cases + 7 cue variants as passes 13–16; dose-response endpoint, not selection.\n")
rq=json.loads((work/"dose_request.json").read_text())
rq["model"]="edited"
json.dump(rq,(work/"dose_request.json").open("w"))
```