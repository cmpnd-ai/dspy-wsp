The program stores formatted answers as strings, so my first field-level summary did not parse them correctly. I’ll use the JSON completions recorded in each model call and inspect the existing diagnostics before changing any weights.

```python
# Hypothesis: reliable field-level comparisons will identify whether workflow-specific evidence readout is the next useful intervention.
import ast
def call_answers(rec):
    a=rec["calls"][0]["completion"]
    return json.loads(a) if isinstance(a,str) else a
print("Completion sample:",str(call_answers(r10[0]))[:1600])
print("Helper files:",[(p.name,p.stat().st_size) for p in WORK.iterdir() if p.is_file() and p.suffix in [".py",".pt"]])
print("First baseline row:",rows[0])
print("Conditional params example:", {str(k):{kk:(tuple(vv.shape) if isinstance(vv,torch.Tensor) else type(vv).__name__) for kk,vv in v.items()} if isinstance(v,dict) else str(type(v)) for k,v in list(conditional_candidates[("soft",.01)]["param"].items())[:1]})
print("Fit/tune summary baseline/best1/best2:",summary_logits(D["logits"],D["istune"]),summary_logits(Lbest,D["istune"]),summary_logits(Ltarget,D["istune"]))
print("Conditional tune table:",conditional_table)
print("QR candidate example:",{k:type(v).__name__ for k,v in qr_candidates[("soft",.001)].items()})
```
