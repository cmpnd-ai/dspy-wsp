The helper returned its results in a different order than I expected, so the summary did not run. I’ll correct the unpacking and repeat the comparison; no weights or results have changed.
```python
# Hypothesis: corrected row alignment will expose the selective versus collateral effects of the gain-4 intervention.
print("Returned types:", type(meta11), type(arr11), type(rowinfo11))
print("Returned keys/samples:", [list(x)[:4] if isinstance(x,dict) else str(x[:1])[:220] for x in (meta11,arr11,rowinfo11)])
import dis
dis.dis(load_read, depth=0)
```
