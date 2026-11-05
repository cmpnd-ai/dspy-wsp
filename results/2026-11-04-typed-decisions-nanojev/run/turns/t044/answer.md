The validated snapshot is intact. I’ll restore it now and locate the prepared dose-response request for comparison.

```python
# Hypothesis: the full validated direction improves decisions beyond partial-strength ablations.
save_edit("scalar.weight", torch.from_numpy(np.load(work/"best_1"/"scalar.weight.npy")).clone())
print("Files:", [p.name for p in work.iterdir() if p.is_file()])
print("Variables:", str(globals().keys())[-6500:])
print("Meta footer:", (work/"fwd_16"/"meta.json").read_text()[-2200:])
```