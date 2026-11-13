The tensor bundle uses a tuple rather than a name-to-tensor mapping. I’ll inspect its layout before proceeding; this is only a file-format check.



```python
# Hypothesis: inspecting the bundle layout will allow an exact, read-only integrity check of the selected edit.
print("best_2 bundle:")
for _k, _v in _saved_best2.items():
    if isinstance(_v, (tuple, list)):
        print(_k, type(_v).__name__, len(_v),
              [(type(z).__name__, tuple(z.shape) if hasattr(z, "shape") else str(z)[:80]) for z in _v[:5]])
    else:
        print(_k, type(_v).__name__, tuple(_v.shape) if hasattr(_v, "shape") else str(_v)[:200])
print("best_1 bundle:")
for _k, _v in _saved_best1.items():
    print(_k, type(_v).__name__, tuple(_v.shape) if hasattr(_v, "shape") else str(_v)[:120])
_probe22 = read_probe_arrays(22)
print("read_probe_arrays result:", type(_probe22).__name__)
if isinstance(_probe22, dict):
    print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in _probe22.items()})
else:
    print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else len(v) if hasattr(v, "__len__") else None) for v in _probe22])
```
