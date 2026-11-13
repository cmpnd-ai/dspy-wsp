The baseline gets 57.9% of questions right. I’ll break that down by question and confusion pair, then inspect representative failures and their actual model requests.

```python
# Hypothesis: systematic option confusions, rather than indiscriminate uncertainty, explain much of the 42.1% error rate.
records = [json.loads(line) for line in open(WORK / "gen_1.jsonl")]
print("Record fields:", list(records[0]))
print("First record (bounded):", json.dumps(records[0], ensure_ascii=False)[:5000])
print("\nQuestion schemas not yet inspected:")
for wf in ["invoice_processing", "security_incidents"]:
    ex = next(ex for ex in examples if ex["inputs"]["workflow"] == wf)
    print(wf, json.dumps(ex["labels"]["questions"], ensure_ascii=False))
lognote("Pass 1: original accuracy 0.579111, SE over 900 items 0.008414; 807 items below perfect. "
        "Records: gen_1.jsonl. No edits.")
```
