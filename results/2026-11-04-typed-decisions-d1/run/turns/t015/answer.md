The row mapping did not match the requested ID order, so the fit stopped before making any edit. I’ll inspect how the read tool orders requests and correct the mapping.

```python
# Hypothesis: read-from requests are emitted in baseline-record order, not the supplied ID order.
print("Requested first",ordered_ids[:5])
print("Baseline-filter first",[r["id"] for r in records if r["id"] in set(ordered_ids[:180])][:5])
for k in ["requests.json","meta.json"]:
    p=work/"fwd_2"/k
    if p.exists(): print(k, p.stat().st_size)
print(os.listdir(work/"fwd_2"))
print([(r["request"],r["question"],r["options"]) for r in meta["rows"][::5][:6]])
```
