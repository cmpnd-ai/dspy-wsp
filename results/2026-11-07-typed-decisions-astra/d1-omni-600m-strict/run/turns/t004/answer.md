The baseline accuracy is 56.56%. I’ll now break down errors on the discovery split and inspect the saved record format so I can distinguish state-reading failures from option-selection biases.
```python
# Hypothesis: repeated confusions on particular questions reveal a systematic decision-head error rather than random uncertainty.
baseline_records = [json.loads(s) for s in (WORK / "gen_1.jsonl").read_text().splitlines()]
print("Record keys:", baseline_records[0].keys())
print(json.dumps(baseline_records[0], ensure_ascii=False)[:9000])
print("Discovery IDs sample:", discovery_ids[:5])
```
