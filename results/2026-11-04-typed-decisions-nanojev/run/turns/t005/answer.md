The baseline accuracy is only 36.4%. I’ll break down its errors by question and inspect representative failures before deciding whether the shared scalar head is responsible.

```python
# Hypothesis: the shared scalar readout systematically confuses particular semantic alternatives.
records = [json.loads(s) for s in (work/"gen_1.jsonl").read_text().splitlines()]
print("Record structure:",str(records[0])[:17000])
print("Head norms:", {n:float(load(n).norm()) for n in ["scalar.weight","scalar.bias","norm.weight","norm.bias","set_output.weight"]})
```
