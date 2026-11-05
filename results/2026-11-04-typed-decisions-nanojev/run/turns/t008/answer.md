I’ll inspect the read-output layout and verify how much of each logit comes from the scalar head versus the set-attention correction. This determines which component can plausibly cause the collapsed predictions.

```python
# Hypothesis: scalar logits, rather than set-attention corrections, dominate the systematic candidate biases.
meta=json.load(open(work/"fwd_2/meta.json"))
print(str(meta)[:14000])
arr={n:np.load(work/f"fwd_2/{n}.npy") for n in ["option_mask","option_logits","head_inputs","path_ends","set_delta","probabilities"]}
for n,a in arr.items(): print(n,a.shape,"range",float(a.min()),float(a.max()))
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":fit_chunks[1]}},open(work/"read_fit1.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nFailure breakdown: agent action 1.3%, customer category 24.4%, invoice duplicate 9.3%; near-fixed candidate preferences across varied states.\nPass 3: original read, next 200 fitting cases; no edits.\n")
```
