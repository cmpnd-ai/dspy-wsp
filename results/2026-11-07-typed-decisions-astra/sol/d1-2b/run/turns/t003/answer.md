The baseline accuracy is 61.3%. I’ll inspect the output format, then break errors down by question and preferred option to distinguish systematic mistakes from isolated cases.

```python
# Hypothesis: a small number of question-specific decision boundaries account for much of the error.
recs = [json.loads(s) for s in (work/"gen_1.jsonl").read_text().splitlines()]
print(json.dumps(recs[0],indent=2)[:18000])
```
