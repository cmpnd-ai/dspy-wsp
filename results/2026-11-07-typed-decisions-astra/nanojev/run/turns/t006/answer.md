I’ll verify how the read arrays map to the decision head and measure the choice correction separately. If the correction is small and the scalar readout ignores distinctions already present in the features, a closed-form readout repair is a plausible targeted intervention.
```python
# Hypothesis: the scalar projection, rather than set attention, causes collapsed option preferences.
f2 = WORK/"fwd_2"
meta2 = json.loads((f2/"meta.json").read_text())
print("Metadata:", json.dumps(meta2, ensure_ascii=False)[:8500])
a2 = {p.stem:np.load(p, mmap_mode="r") for p in f2.glob("*.npy")}
print("Arrays:", {k:v.shape for k,v in a2.items()})
X2 = np.asarray(a2["head_inputs"])
raw2 = X2 @ head0["scalar.weight"].numpy().reshape(-1) + head0["scalar.bias"].item()
mask2 = np.asarray(a2["option_mask"]).astype(bool)
print("Scalar raw percentiles:",np.percentile(raw2[mask2], [0,5,25,50,75,95,100]).round(4))
print("Set delta percentiles:",np.percentile(a2["set_delta"][mask2],[0,5,25,50,75,95,100]).round(5))
print("Raw + correction vs logits, maxdiff:", np.max(np.abs((raw2 + a2["set_delta"] - a2["option_logits"])[mask2])))
print("Feature variability: mean L2 std per coord",round(float(np.linalg.norm(X2[mask2].std(axis=0))),4))
# Get the next 200 fitting records, filling in invoice and security workflows.
read3_ids = splits["fit"][200:400]
(WORK/"read3_request.json").write_text(json.dumps({"type":"read","model":"original","from":{"pass":1,"ids":read3_ids}}))
note("Planned forward 3: original READ fit IDs 200:400 (exact read3_request.json), 1000 questions. Edits: none. Complete evidence across all workflows before deriving a readout direction.")
```
