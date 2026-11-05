I’ll create a baseline request for all 900 practice cases. This will show which workflows and question types account for the errors.

```python
# Hypothesis: measuring all workflows avoids selecting a repair from a small, unrepresentative sample.
json.dump({"type":"decide","model":"original","problems":[x["id"] for x in data]}, open(work/"baseline_request.json","w"))
```
