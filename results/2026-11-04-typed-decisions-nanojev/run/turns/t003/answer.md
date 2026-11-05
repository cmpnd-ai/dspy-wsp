I’ll run the original model on all 900 practice records to identify which questions fail most often and inspect the corresponding requests and probabilities.

```python
# Hypothesis: baseline errors cluster by question or workflow, revealing a specific readout failure.
baseline_request = {"type":"decide","model":"original","problems":[e["id"] for e in examples]}
json.dump(baseline_request,open(work/"baseline_request.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nPass 1: original model, all 900 practice records; no edits. Baseline diagnosis only.\n")
```
