The unedited model gets 61.3% of questions right. I’ll now break that down by workflow, question, and mistaken option, and inspect the recorded requests so that any repair addresses what the model actually sees.
```python
# Hypothesis: a few systematic option confusions account for a large share of the 38.7% baseline error.
baseline = [json.loads(s) for s in (WORK/"gen_1.jsonl").read_text().splitlines() if s.strip()]
print("Record keys:", baseline[0].keys())
print(json.dumps(baseline[0], indent=2)[:9000])
note("Pass 1 baseline: 900 records, mean accuracy 0.613333, item SE 0.008248; 798/900 items imperfect. Output: gen_1.jsonl. No edits.")
```
