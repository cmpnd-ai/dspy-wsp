The original model scores 48%. I’ll break that down by question and inspect actual failed calls to distinguish rule-following errors from decision-head bias.

```python
# Hypothesis: the scalar readout systematically misranks particular answer types despite useful backbone representations.
records = [json.loads(s) for s in open(work/"gen_1.jsonl")]
print("Record structure:", str(records[0])[:9000])
```
