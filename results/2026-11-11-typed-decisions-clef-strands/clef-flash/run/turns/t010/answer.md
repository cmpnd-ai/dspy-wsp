Next I’ll collect the remaining fitting records and the tuning split. I’ll also measure each logit component’s option-to-option variation; a large constant contribution cannot affect which option wins.

```python
# Hypothesis: the scorer has useful state-sensitive activations, but its output weights fail to convert them into correct option margins.
print("Meta keys:", list(meta2.keys()))
for k,v in meta2.items():
    if k!="rows":
        print(k, str(v)[:1400])
print("First requested IDs:", read_chunks[0][:6])
print("Request mapping:", [(i, byid[i]["inputs"]["workflow"]) for i in read_chunks[0][:6]])
valid2 = mask2
def center_options(t, m):
    mm = m.to(t.dtype)
    if t.ndim==3:
        mm = mm.unsqueeze(-1)
    return (t-(t*mm).sum(1,keepdim=True)/mm.sum(1,keepdim=True).clamp_min(1))*mm
components2 = {
    "prior":torch.tensor(np.array(A2["prior"])),
    "cosine":gate*jscale*torch.tensor(np.array(A2["cosine"])),
    "residual":gate*(H2@wout+bout)
}
for k,t in components2.items():
    print(k,"centered RMS",center_options(t,valid2)[valid2].square().mean().sqrt().item())
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":read_chunks[2]}},
          (WORK/"read_fit_tune.json").open("w"))
note("## Planned pass 4\nExact edits: none. READ original on remaining 100 fit plus 100 tune records (read_fit_tune.json, 1000 questions). Will solve a regularized, within-question-centered least-squares correction to residual_scorer.3.weight using fit only; tune chooses regularization/strength. This is a direct closed-form solve, not iterative training.")
```
