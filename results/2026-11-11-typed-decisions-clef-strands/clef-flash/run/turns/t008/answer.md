The read tool returned the exact 4,096-dimensional scorer features. I’ll reconstruct the final logits locally and separate the lexical prior, cosine term, and residual scorer. That checks whether the final readout is a plausible repair site.

```python
# Hypothesis: incorrect option margins can be localized to the existing residual scorer without changing evidence routing.
torch.set_num_threads(min(8, os.cpu_count() or 4))
meta2 = json.loads((WORK/"fwd_2/meta.json").read_text())
print("Meta type:", type(meta2))
print("Meta preview:", json.dumps(meta2, ensure_ascii=False)[:4200])
A2 = {p.stem:np.load(p, mmap_mode="r") for p in (WORK/"fwd_2").glob("*.npy")}
W0 = load_original("residual_scorer.0.weight")
b0 = load_original("residual_scorer.0.bias")
wout = load_original("residual_scorer.3.weight").reshape(-1)
bout = load_original("residual_scorer.3.bias")
gate = load_original("residual_gate").sigmoid().item()
jscale = load_original("joint_logit_scale").clamp(max=math.log(100)).exp().item()
pscale = load_original("prior_logit_scale").clamp(max=math.log(100)).exp().item()
H2 = F.gelu(F.linear(torch.from_numpy(np.array(A2["head_inputs"])), W0, b0))
pred2 = torch.from_numpy(np.array(A2["prior"])) + gate*(jscale*torch.from_numpy(np.array(A2["cosine"])) + H2@wout+bout)
mask2 = torch.from_numpy(np.array(A2["option_mask"])).bool()
err2 = (pred2-torch.from_numpy(np.array(A2["option_logits"])))[mask2]
print("Scalar components:", dict(gate=gate,joint_scale=jscale,prior_scale=pscale))
print("Reconstruction error max/mean:", err2.abs().max().item(),err2.abs().mean().item())
print("Output weights norm/max:", wout.norm().item(),wout.abs().max().item())
print("Activation mean/std/range:", H2[mask2].mean().item(), H2[mask2].std().item(), H2[mask2].min().item(), H2[mask2].max().item())
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":read_chunks[1]}},
          (WORK/"read_fit_2.json").open("w"))
note(f"## Pass 2 result\nRead 200 fit records, 1000 questions. Scorer reconstructed with max logit error {err2.abs().max().item():.7g}; residual gate={gate:.6g}, joint cosine scale={jscale:.6g}, prior scale={pscale:.6g}.\n## Planned pass 3\nExact edits: none. READ original on next 200 fit IDs (read_fit_2.json); no tune/test labels used.")
```
