Before designing another edit, I’ll check where the existing repairs helped or hurt on discovery cases. This will show whether gating the payment and violation readouts is worth testing.
```python
# Hypothesis: remaining collateral errors are concentrated outside the intended workflow/questions and can be isolated by gates.
print("ANALYSIS HELPERS")
for k,v in list(globals().items()):
    if isinstance(v, types.FunctionType) and not k.startswith("_"):
        try: print(k, str(inspect.signature(v))[:180])
        except Exception: pass
print("\nARRAYS")
print([(k,v.shape) for k,v in list(globals().items()) if isinstance(v,np.ndarray) and not k.startswith("_")])
print("\nMETA3 SAMPLE", json.dumps(meta3["rows"][0])[:1300])
print("FILES", meta3["files"])
print("CONVENTIONS", meta3["conventions"])
print("\nIDENTIFIER VARIABLES")
print([(k, type(v).__name__, len(v)) for k,v in list(globals().items())
       if not k.startswith("_") and isinstance(v,(dict,list)) and any(s in k.lower() for s in ["ids","record","example","baseline","target"])])
```
