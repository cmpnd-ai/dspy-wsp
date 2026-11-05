I’ll check the read-request format so the full-strength run matches the earlier diagnostics exactly.

```python
# Hypothesis: identical requests isolate the scalar direction from input differences.
print((work/"panel_request.json").read_text()[:180])
print("install function:")
import inspect
print(inspect.getsource(install_job))
print("read_batch:", inspect.getsource(read_batch)[:2200])
print("eval_dw:", inspect.getsource(eval_dw)[:2400])
```