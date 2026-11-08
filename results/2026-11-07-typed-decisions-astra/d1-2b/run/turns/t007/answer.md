I’ll inspect the read-file definitions before doing any algebra, especially whether “head inputs” are normalized residuals or pre-normalization vectors. Getting that distinction right is essential because the last MLP’s write is RMS-normalized before readout.
```python
# Hypothesis: the read files expose enough final-head structure to reconstruct and causally modify an option-specific direction.
meta2 = json.loads((WORK/"fwd_2"/"meta.json").read_text())
print("Meta type/keys:", type(meta2), list(meta2) if isinstance(meta2,dict) else len(meta2))
print(json.dumps(meta2,indent=2)[:8500])
print("Files:", [p.name for p in (WORK/"fwd_2").iterdir()])
for name in ["head_inputs","option_rows","hidden_mean","option_logits","probabilities","option_mask"]:
    a=np.load(WORK/"fwd_2"/(name+".npy"), mmap_mode="r")
    print(name, a.shape, a.dtype, "range",float(a.min()),float(a.max()))
print("Torch threads:",torch.get_num_threads())
```
