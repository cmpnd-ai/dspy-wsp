I’ll verify the option names and request alignment before computing contrasts. This avoids mixing up padded columns or relying on request order.
```python
# Hypothesis: matching complete requests and named options yields an unambiguous causal audit.
audit_sources23={
    6:(arr6,meta6,json.loads((work/"deadline_counterfactuals.json").read_text())["requests"],synthetic_tags),
    9:(arr9,meta9,json.loads((work/"agent_counterfactuals.json").read_text())["requests"],agent_tags),
    14:(arr14,meta14,json.loads((work/"security_counterfactuals.json").read_text())["requests"],security_cf_tags),
}
stored23=json.loads((work/"fwd_23"/"requests.json").read_text())
print("STORED REQUEST TYPE",type(stored23).__name__)
if isinstance(stored23,dict):
    print("STORED KEYS",list(stored23))
else:
    print("REQUEST KEYS",list(stored23[0]))
for family23,start23 in [("payment",0),("violation",320),("security",640)]:
    print(family23,[(r["question"],r["options"]) for r in meta23["rows"][start23:start23+5]])
print("SOURCE REQUEST TYPE",type(json.loads((work/"fwd_6"/"requests.json").read_text())).__name__)
```
