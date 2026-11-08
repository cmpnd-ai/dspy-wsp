I can measure the last branch’s contribution more directly: apply two different nonuniform norm perturbations, read the same answer states, then account for RMS normalization. This will identify gain directions by their causal effect, rather than assuming the branch carries the full residual signal.

```python
# Hypothesis: controlled nonuniform norm perturbations reveal the last branch's causal contribution per dimension.
rngnp=np.random.default_rng(851)
d1=torch.tensor(rngnp.choice([-1.,1.],2048),dtype=torch.float32)*g0
d2=torch.tensor(rngnp.choice([-1.,1.],2048),dtype=torch.float32)*g0
save_edit(gn,g0+d1)
note("Pass 3 READ: causal probe 1, last post_ffn_norm = g0 + random-sign*g0 (seed 851), same 200 fit requests as pass 2. No other edits.")
req=json.load(open(work/"read_request.json")); req["model"]="edited";req.pop("layers",None)
json.dump(req,open(work/"probe_request.json","w"))
print("probe norm change",torch.linalg.vector_norm(d1).item(), "changed",int((d1!=0).sum()))
```
