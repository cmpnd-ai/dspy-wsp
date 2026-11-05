I’ll check the numerical decomposition and prepare the last fitting batch. If set-attention corrections are negligible, I can repair the scalar readout directly without altering the backbone.

```python
# Hypothesis: the failure is localized to the scalar projection; quantify the competing set correction.
print("Metadata fields:",{k:v for k,v in meta.items() if k!="rows"})
for n,a in arr.items(): print(n,a.shape, "min/max",float(a.min()),float(a.max()))
w0=load_original("scalar.weight").numpy().reshape(-1).copy()
b0=float(load_original("scalar.bias")[0])
calc=arr["head_inputs"]@w0+b0+arr["set_delta"]
mask=arr["option_mask"].astype(bool)
print("Scalar reconstruction errors on mask:",np.quantile(np.abs(calc-arr["option_logits"])[mask],[0,.5,.99,1]))
print("Set correction magnitude:",np.quantile(abs(arr["set_delta"][mask]),[.5,.9,1]))
print("First boolean inputs/logits:",[(i,arr["option_logits"][i,:2].tolist(),np.linalg.norm(arr["head_inputs"][i,:2],axis=1).tolist()) for i in range(5) if meta["rows"][i]["type"]=="noul"])
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":fit_chunks[2]}},open(work/"read_fit2.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nPass 4 planned: original read, final 200 fitting cases; no edits. All candidate edits will be closed-form scalar projections, not backbone changes.\n")
```
