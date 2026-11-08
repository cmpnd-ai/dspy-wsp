I’ll check exactly where the exported “head inputs” sit in the network. That determines whether a targeted edit can change how the scorer reads an existing distinction without changing the encoder.
```python
# Hypothesis: the exported option features provide a localized site for a contrastive, non-iterative readout repair.
F2 = WORK / "fwd_2"
meta2 = json.loads((F2/"meta.json").read_text())
print(json.dumps(meta2,ensure_ascii=False)[:7500])
arr2 = {p.stem: np.load(p, mmap_mode="r") for p in F2.glob("*.npy")}
for k,a in arr2.items():
    print(k,a.shape,a.dtype)
print("Folder contents:",[p.name for p in F2.iterdir()])
```
