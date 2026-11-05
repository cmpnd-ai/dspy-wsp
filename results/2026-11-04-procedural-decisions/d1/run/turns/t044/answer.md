The helper’s source is unavailable in this notebook, so I’ll inspect its arguments instead and restore C directly. No weights changed in the failed inspection.

```python
# Hypothesis: we can test localized scorer directions without changing the encoder.
print("measure signature",inspect.signature(measure))
print("Lists",[(k,len(v)) for k,v in list(globals().items()) if isinstance(v,list) and len(v)>50])
save_edit("head.scorer.3.weight",candidateC)
req=json.loads((work/"read_fit.json").read_text())
print("request",str(req)[:500])
print("C relative norm",float(torch.linalg.vector_norm(wC-w0)/torch.linalg.vector_norm(w0)))
print("measure",measure.__code__.co_varnames)
```
