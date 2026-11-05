The read panel uses a different ordering from the ID list. I’ll align it using the saved request order before scoring.

```python
# Hypothesis: alignment by saved request order yields valid dose-response measurements.
print("panel_order", str(panel_order)[:1100])
print("panel request count",len(panel_requests))
print("job installation bytecode")
import dis
dis.dis(install_job)
```