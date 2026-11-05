The baseline accuracy is 51.6%. I’ll break down errors by question and inspect failed records to distinguish missing reasoning from a head that misreads useful features.

```python
# Hypothesis: systematic option preferences, rather than random failures, explain much of the baseline error.
base = [json.loads(s) for s in (work/"gen_1.jsonl").read_text().splitlines()]
print(json.dumps(base[0],indent=2)[:12000])
```
