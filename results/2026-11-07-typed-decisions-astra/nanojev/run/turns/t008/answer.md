I’ll resolve the head-feature layout, especially the special boolean path, and measure how much set attention changes relative logits. I’ll also request the remaining 50 fitting records and all 150 development records; the reserved test records remain untouched by fitting.
```python
# Hypothesis: boolean false is a fixed zero logit, and the apparent reconstruction error comes from its non-path slot.
print("Metadata fields besides rows:", {k:v for k,v in meta2.items() if k!="rows"})
print("Example rows with raw/head values:")
for j in range(5):
    m=meta2["rows"][j]; n=len(m["options"])
    print(m, "mask",mask2[j,:n].tolist(),"logit",a2["option_logits"][j,:n].round(4).tolist(),
          "raw",raw2[j,:n].round(4).tolist(),"delta",a2["set_delta"][j,:n].round(4).tolist(),
          "head norm",np.linalg.norm(X2[j,:n],axis=-1).round(3).tolist())
valid_path2=mask2.copy()
for j,m in enumerate(meta2["rows"]):
    if m["type"]=="noul": valid_path2[j,0]=False
print("Scalar reconstruction max (real paths):",float(np.max(np.abs((raw2+a2["set_delta"]-a2["option_logits"])[valid_path2]))))
print("head_inputs vs path_ends max:",float(np.max(np.abs(a2["head_inputs"][valid_path2]-a2["path_ends"][valid_path2]))))
ends=torch.tensor(np.array(a2["path_ends"][:5]))
ln_ends=torch.nn.functional.layer_norm(ends,(1024,),head0["norm.weight"],head0["norm.bias"],eps=1e-5)
print("LN(path_ends) vs head_inputs max:",float((ln_ends-torch.tensor(np.array(a2["head_inputs"][:5]))).abs().max()))
rel_delta=[]
for j,m in enumerate(meta2["rows"]):
    if m["type"]=="choice":
        d=np.asarray(a2["set_delta"][j,:len(m["options"])])
        rel_delta.append(float(d.max()-d.min()))
print("Choice correction within-question range quantiles:",np.percentile(rel_delta,[0,50,90,100]).round(6))
SNAP=pathlib.Path("/Users/dbreunig/.cache/huggingface/hub/models--C-Tianyu--NanoJev/snapshots/047b927b30882a1138fc504821b82ac145a4b81a")
print("Snapshot files:",[p.name for p in SNAP.iterdir()])
for p in SNAP.glob("*config*.json"):
    if "tokenizer" not in p.name:
        print(p.name,p.read_text()[:5500])
read4_ids=splits["fit"][400:]+splits["dev"]
(WORK/"read4_request.json").write_text(json.dumps({"type":"read","model":"original","from":{"pass":1,"ids":read4_ids}}))
note("Planned forward 4: original READ last50 fit IDs + all150 dev IDs, exact read4_request.json. Edits: none. Boolean false fixed-logit slot must be excluded from scalar reconstruction. Head decomposition and closed-form design follow.")
```
