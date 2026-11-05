The tool emits requests in baseline-record order. I’ll map each saved request back to its exact JSON prompt, then rerun the closed-form calculation with verified labels.

```python
# Hypothesis: exact prompt matching safely aligns contextual features and gold option distributions.
reqs=json.load(open(work/"fwd_2/requests.json"))
print(type(reqs),str(reqs)[:300])
```
