I’ll finish the integrity checks, save a copy of the exact submitted tensor, and run the final paired validation. This repeats the already-confirmed C candidate; it is not a new independent validation sample.

```python
# Hypothesis: the exact restored C artifact retains its previously confirmed held-out gain.
hr0=np.load(work/"fwd_8/head_inputs.npy")
hr1=np.load(work/"fwd_24/head_inputs.npy")
lr1=np.load(work/"fwd_24/option_logits.npy")
print("Reserve head input difference",np.nanmax(np.abs(hr0-hr1)))
print("Reserve replay error",np.nanmax(np.abs(np.einsum("qod,d->qo",Ar,wC)+bias-lr1)))
print("Current edits",edited())
print("Exact C match",torch.equal(load("head.scorer.3.weight"),torch.as_tensor(candidateC)))
torch.save(load("head.scorer.3.weight"),work/"final_confirmed_C.pt")
with (work/"notes.md").open("a") as f:
    f.write("\nPass 24 reserve head verification complete. Final edit: head.scorer.3.weight only, full candidate C; all other tensors untouched. Closed-form ridge readout, lambda .1, centered option activations/one-hot targets, 840 practice cases, no gradients. Final pass 25 repeats paired validation with unchanged C to verify restored artifact. Earlier validation C .54333 -> .73000, gain .18667 SE .02685. This repeat is not independent evidence.\n")
```
