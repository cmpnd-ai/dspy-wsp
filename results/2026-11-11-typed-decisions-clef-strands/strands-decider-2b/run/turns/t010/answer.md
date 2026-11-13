I’ll verify the exact head computation and inspect which dimensions dominate its inputs. Large shared activation directions could obscure the smaller directions that distinguish state evidence and option meaning.

```python
# Hypothesis: a small number of common activation directions dominate the head, while useful option contrasts remain available.
torch.set_num_threads(min(6, os.cpu_count() or 6))
print("Metadata keys:", list(meta2))
for key, val in meta2.items():
    if key != "rows":
        print(key, str(val)[:2500])
d2_raw = torch.from_numpy(np.load(WORK/"fwd_2"/"decide.npy")[:,0,:].copy())
o2_raw = torch.from_numpy(np.load(WORK/"fwd_2"/"head_inputs.npy"))
mask2 = torch.from_numpy(np.load(WORK/"fwd_2"/"option_mask.npy"))
L2 = torch.from_numpy(np.load(WORK/"fwd_2"/"option_logits.npy"))
def ln0(x):
    return torch.nn.functional.layer_norm(x, (2048,), H0["norm.weight"], H0["norm.bias"], 1e-5)
d2 = ln0(d2_raw)
o2 = ln0(o2_raw.nan_to_num())
q2 = torch.nn.functional.linear(d2,H0["q.weight"],H0["q.bias"])
k2 = torch.nn.functional.linear(o2,H0["k.weight"],H0["k.bias"])
predL2 = torch.einsum("nkd,nd->nk",k2,q2)/math.sqrt(768)
print("Max exact-logit error:", (predL2[mask2]-L2[mask2]).abs().max().item())
for name, x in [("decision",d2),("option",o2[mask2])]:
    energy = x.square().mean(0)
    top = energy.topk(10)
    print(name,"top dimensions",list(zip(top.indices.tolist(),np.round(top.values.numpy(),3).tolist())),
          "energy share",top.values.sum().item()/energy.sum().item(),
          "mean norm",x.mean(0).norm().item(),"centered RMS",x.std(0).square().mean().sqrt().item())
lognote("Head LayerNorm scale is all ones, bias all zeros. No head edits. "
        f"Exact cached head reproduction max error {(predL2[mask2]-L2[mask2]).abs().max().item():.6g}.")
lognote("## Before pass 4\nExact edit: none. READ original fit_c, final 180 fitting items, 900 questions. "
        "Exact IDs in read_fit_c.json. All 540 fitting cases will then have head-input coverage.")
```
