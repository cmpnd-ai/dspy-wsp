The set correction is small, while the scalar head produces positive logits for every boolean. I’ll construct within-question contrasts (including the fixed false logit of zero) and solve once for an additive evidence direction. This is a closed-form least-squares edit, not backbone training.

```python
# Hypothesis: normalized EOS features contain discriminative evidence that a small additive scalar direction can expose.
print("Metadata keys:",meta.keys())
for k,v in meta.items():
    if k!="rows": print(k,str(v)[:1500])
H=arr["head_inputs"].astype(np.float64)
W0=load_original("scalar.weight").numpy().reshape(-1).astype(np.float64)
b0=float(load_original("scalar.bias")[0])
raw=H@W0+b0
for q in stats:
    inds=[i for i,r in enumerate(meta["rows"]) if r["question"]==q]
    i=inds[0]
    print(q, "head norms",np.linalg.norm(H[i],axis=1).round(3),"raw",raw[i].round(4),
          "logits",arr["option_logits"][i].round(4),"delta",arr["set_delta"][i].round(4))
print("fit_ids first",fit_ids[:4])
```
