I’ll inspect the saved files directly, avoiding the shadowed namespace helpers.

```python
# Hypothesis: on-disk artifacts suffice to recover the validated edit and diagnostics.
print([p.name for p in work.iterdir() if p.is_dir()])
print((work/"fwd_16"/"meta.json").read_text()[:3500])
print("best files", [str(p) for p in (work/"best_1").rglob("*")][:10])
```