The baseline is 70.13% correct. I’ll break that down by question and inspect the head implementation so we can distinguish option-selection errors from calibration issues.


```python
# Hypothesis: systematic option confusions, rather than global confidence, account for most baseline failures.
base = [json.loads(s) for s in (WORK/"gen_1.jsonl").read_text().splitlines()]
print("Record keys:", base[0].keys())
print("First record:", json.dumps(base[0], ensure_ascii=False)[:8500])
source = (MODEL/"joint_schema_model.py").read_text()
print("Config:", (MODEL/"joint_head_config.json").read_text())
print("Head source structure:")
for i,line in enumerate(source.splitlines()):
    if line.lstrip().startswith(("class ", "def ")):
        print(i+1, line)
note("## Pass 1 result\nOriginal practice accuracy 0.701333; item SE 0.007763, n=900; 712 items below perfect. No program or metric failures. All subsequent mechanism analysis uses these practice records only.")
```
