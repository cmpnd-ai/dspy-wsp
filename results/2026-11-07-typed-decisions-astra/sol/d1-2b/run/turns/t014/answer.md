The reconstruction reproduces the probes’ logits closely, but the candidate fit accuracy looks suspiciously low. I’ll check row alignment and baseline reconstruction before trusting any correction.

```python
# Hypothesis: an ordering or normalization mismatch, rather than the gain edit, explains the low apparent fit accuracy.
print("base accuracy",np.mean(np.nanargmax(L,1)==target),"reconstructed",np.mean(np.nanargmax(logits_gain(np.zeros(2048)),1)==target))
print("Z0 rms",np.quantile(np.sqrt((Z0**2).mean(1)),[0,.5,1]))
requests=json.load(open(fdir/"requests.json"))
print(type(requests),str(requests)[:1200])
print("readfit first ids", [e['id'] for e in readfit[:5]])
print("read meta first",meta["rows"][:2])
print("expected feedback",next(r['feedback'] for r in recs if r['id']==readfit[0]['id']))
print("first logits",L[:5], "targets",target[:5])
```
