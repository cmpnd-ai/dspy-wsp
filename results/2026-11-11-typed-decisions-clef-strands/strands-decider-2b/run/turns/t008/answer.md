The read results are cached and inexpensive. I’ll inspect their layout and reproduce the original logits before attempting any edit; this guards against fitting a repair to the wrong representation.

```python
# Hypothesis: exact head reproduction will isolate readout errors from backbone or prompt-encoding effects.
meta2 = json.load(open(WORK / "fwd_2" / "meta.json"))
print(json.dumps(meta2, ensure_ascii=False)[:7500])
print("Head tensor statistics:")
head_names = ["norm.weight","norm.bias","q.weight","q.bias","k.weight","k.bias"]
H0 = {n: load_original(n).clone() for n in head_names}
for n,t in H0.items():
    print(n, tuple(t.shape), "norm", round(t.norm().item(),4), "std", round(t.std().item(),5))
for f in ["head_inputs","decide","option_mask","option_logits","probabilities"]:
    a = np.load(WORK/"fwd_2"/(f+".npy"), mmap_mode="r")
    print(f, a.shape, a.dtype, "first row summary", np.asarray(a[0]).reshape(-1)[:8])
lognote("## Before pass 3\nExact edit: none. READ original fit_b, next disjoint 180 fit items, 900 questions. "
        "Exact IDs in read_fit_b.json. Pass 2 outputs in fwd_2; inspect array metadata before deriving edits.")
```
