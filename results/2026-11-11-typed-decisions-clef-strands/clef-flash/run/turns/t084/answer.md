The saved file is a bundle containing tensors and metadata; my check mistakenly treated the metadata keys as tensor names. No weights changed. I’ll unpack it correctly and load the probe arrays.


```python
# Hypothesis: unpacking the saved tensor bundle will confirm that best_2 is still deployed exactly.
_best2_tensors = _saved_best2["tensors"]
print("Best_1 bundle keys:", list(_saved_best1))
_best1_tensors = _saved_best1.get("tensors", _saved_best1)
print("Exact saved/edit equality:", {
    n: bool(torch.equal(load(n), v.to(torch.bfloat16).float()))
    for n, v in _best2_tensors.items()
})
print("Probe families and levels:")
for _fam in sorted({m["family"] for m in held_probe_meta}):
    print(_fam, sorted({str(m["level"]) for m in held_probe_meta if m["family"] == _fam}))
print("Fingerprint:", edit_fingerprint())
_probe22 = read_probe_arrays(22)
print("read_probe_arrays result:", type(_probe22).__name__)
if isinstance(_probe22, dict):
    print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in _probe22.items()})
else:
    print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else len(v) if hasattr(v, "__len__") else None) for v in _probe22])
```
