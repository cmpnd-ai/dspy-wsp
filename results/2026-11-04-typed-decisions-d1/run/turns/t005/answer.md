The baseline accuracy is 56.6%. I’ll inspect the saved record format, then summarize errors by workflow and question.

```python
# Hypothesis: systematic option confusions account for much of the 43.4% question error rate.
records = [json.loads(s) for s in (work/"gen_1.jsonl").read_text().splitlines()]
print(json.dumps(records[0], indent=2)[:9500])
```
