I’ll inspect the read-output layout and verify exactly how the scalar head reconstructs the recorded logits. This avoids fitting against the wrong feature normalization or overlooking the choice-set correction.

```python
# Hypothesis: the recorded head inputs isolate the scalar readout and allow exact causal predictions of head edits.
meta=json.load(open(work/"fwd_2/meta.json"))
print(str(meta)[:7000])
arr={p.stem:np.load(p) for p in (work/"fwd_2").glob("*.npy")}
for n in ["scalar.weight","scalar.bias","norm.weight","norm.bias","set_output.weight"]:
    t=load(n); print(n,tuple(t.shape),"norm",t.norm().item(),"range",t.min().item(),t.max().item())
print("Arrays", {k:(v.shape, float(np.nanmin(v)),float(np.nanmax(v))) for k,v in arr.items()})
```
