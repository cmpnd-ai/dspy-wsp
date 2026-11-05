I’ll prepare a baseline request for all 900 practice cases. The per-question results will show which confusions are large enough to target.

```python
# Hypothesis: a full baseline will reveal systematic errors rather than isolated difficult cases.
baseline_request = {"type":"decide","model":"original","problems":[x["id"] for x in examples]}
json.dump(baseline_request, open(work/"baseline_request.json","w"))
with notes.open("a") as f:
    f.write("\nPass 1 planned: original weights, all 900 practice examples; no edits.\n")
```
