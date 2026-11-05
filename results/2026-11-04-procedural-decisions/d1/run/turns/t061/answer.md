I’ll verify the second partition and check the 60-case reserve set at the head boundary. After that, I’ll repeat validation with C unchanged as the final submission check.

```python
# Hypothesis: the same final-readout mechanism holds across all practice partitions.
ht0=np.load(work/"fwd_3/head_inputs.npy")
ht1=np.load(work/"fwd_23/head_inputs.npy")
lt1=np.load(work/"fwd_23/option_logits.npy")
print("Second partition head input difference",np.nanmax(np.abs(ht0-ht1)))
print("Second partition replay error",np.nanmax(np.abs(np.einsum("qod,d->qo",At,wC)+bias-lt1)))
req=json.loads((work/"read_reserve.json").read_text())
req["model"]="edited"
(work/"verify_reserve.json").write_text(json.dumps(req))
with (work/"notes.md").open("a") as f:
    f.write("\nPass 23 C second-partition edited read completed; upstream features checked against original. Next pass 24 READ edited 60 reserve cases (excluded from C fit). Full C only final scorer readout remains installed. Then final unchanged-C validation confirmation.\n")
```
