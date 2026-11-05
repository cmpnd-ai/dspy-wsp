A surviving variable appears to shadow a Python built-in. I’ll use explicit built-ins to inspect the session safely.

```python
# Hypothesis: saved snapshots and diagnostic variables identify the exact validated repair.
import builtins as bi
print([(k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "")
       for k, v in bi.list(globals().items())
       if not k.startswith("_") and k not in ["NAMES", "SHAPES"]][-110:])
print("Snapshots:", [p.name for p in work.iterdir() if p.is_dir()])
print((work/"fwd_16"/"meta.json").read_text()[:2600])
```