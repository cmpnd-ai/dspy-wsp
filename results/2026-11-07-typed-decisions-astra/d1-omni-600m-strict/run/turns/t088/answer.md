I’ll break down the discovery-set changes by workflow and question. If the payment and violation readouts are helping their intended questions but causing errors elsewhere, that gives us a specific gating hypothesis to test.
```python
# Hypothesis: readout gains and collateral errors separate by question, supporting a more selective edit.
print("CACHED PASS VARIABLES")
print([(k, type(v).__name__) for k,v in list(globals().items())
       if not k.startswith("_") and any(k.startswith(p) for p in ["arr","info","a2","a3","a7","a8","a10","a18","r2","r3","r10"])])
print("EXAMPLE SCHEMA", examples[0].keys())
print("READ_HELPER_NAMES", load_read.__code__.co_names)
print("SUMMARY_HELPER_NAMES", read_summary.__code__.co_names)
discovery22_base = load_read(3, read_ids_b)
print("LOAD RESULT", type(discovery22_base).__name__,
      [(type(z).__name__, list(z)[:8] if isinstance(z,dict) else len(z)) for z in discovery22_base])
```
