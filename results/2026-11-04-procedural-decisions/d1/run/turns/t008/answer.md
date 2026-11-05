I’ll inspect the read-output metadata and scorer weights, then verify that I can reproduce the raw logits from the supplied features. This checks exactly where a closed-form repair can act.

```python
# Hypothesis: head_inputs are the post-transformer option vectors, enabling an exact scorer-only causal intervention.
meta=json.load(open(work/"fwd_2/meta.json"))
print(str(meta)[:6500])
H=np.load(work/"fwd_2/head_inputs.npy")
L=np.load(work/"fwd_2/option_logits.npy")
M=np.load(work/"fwd_2/option_mask.npy")
print("shapes",H.shape,L.shape,M.shape)
print("scorer", [(n,tuple(load(n).shape)) for n in NAMES if n.startswith("head.scorer")])
```
