I’ll measure how much the original logits depend on query variation versus option representations. I’ll also prepare an untouched development read, so any closed-form correction can be judged beyond the examples used to derive it.

```python
# Hypothesis: the nearly constant decision query makes the key projection the main bottleneck for reading option-specific evidence.
temps = {"choice":0.863,"noul":1.070,"score":1.834}
def load_read(passnum, ids):
    root = WORK/f"fwd_{passnum}"
    meta = json.load(open(root/"meta.json"))["rows"]
    mask = torch.from_numpy(np.load(root/"option_mask.npy"))
    d_raw = torch.from_numpy(np.load(root/"decide.npy")[:,0,:].copy())
    o_raw = torch.from_numpy(np.load(root/"head_inputs.npy"))
    d = ln0(d_raw)
    o = ln0(o_raw.nan_to_num())
    logits = torch.from_numpy(np.load(root/"option_logits.npy")).nan_to_num()
    probabilities = torch.from_numpy(np.load(root/"probabilities.npy")).nan_to_num()
    rowids = [ids[m["request"]] for m in meta]
    group = [byid[i]["inputs"]["workflow"]+"/"+m["question"] for i,m in zip(rowids,meta)]
    gold = torch.zeros_like(logits)
    for n,(i,m) in enumerate(zip(rowids,meta)):
        gg = byid[i]["labels"]["gold"][m["question"]]["probabilities"]
        gold[n,:len(m["options"])] = torch.tensor([gg[opt] for opt in m["options"]])
    return {"d":d,"o":o,"mask":mask,"L":logits,"p":probabilities,"gold":gold,
            "temp":torch.tensor([temps[m["type"]] for m in meta]),
            "ids":rowids,"group":group,"meta":meta}
def concat_data(parts):
    out={}
    for key in parts[0]:
        out[key] = torch.cat([p[key] for p in parts]) if torch.is_tensor(parts[0][key]) else sum([p[key] for p in parts],[])
    return out
fit_parts = [load_read(j,fit_batches[j-2]) for j in [2,3,4]]
fit = concat_data(fit_parts)
del fit_parts
def center_options(x, mask):
    if x.ndim==3:
        return (x-(x*mask[:,:,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None])*mask[:,:,None]
    return (x-(x*mask).sum(1,keepdim=True)/mask.sum(1)[:,None])*mask
def head_q(d, H=H0):
    return torch.nn.functional.linear(d,H["q.weight"],H["q.bias"])
def head_k(o, H=H0):
    return torch.nn.functional.linear(o,H["k.weight"],H["k.bias"])
fit["Q"] = head_q(fit["d"])
qmean = fit["Q"].mean(0)
u = qmean/qmean.norm()
fit["g"] = fit["Q"]@u/math.sqrt(768)
constL = torch.einsum("nkd,d->nk",head_k(fit["o"]),qmean)/math.sqrt(768)
deltaL = center_options(fit["L"]-constL,fit["mask"])
print("Fit query: mean norm",qmean.norm().item(),"centered RMS norm",(fit["Q"]-qmean).square().sum(1).mean().sqrt().item())
print("Query-axis factor min/mean/max",fit["g"].min().item(),fit["g"].mean().item(),fit["g"].max().item())
print("Contextual query contribution to centered logits: RMS",deltaL[fit["mask"]].square().mean().sqrt().item())
print("Original centered logits RMS",center_options(fit["L"],fit["mask"])[fit["mask"]].square().mean().sqrt().item())
pred0 = fit["L"].masked_fill(~fit["mask"],-1e9).argmax(1)
predconst = constL.masked_fill(~fit["mask"],-1e9).argmax(1)
print("Replacing all queries by mean changes predictions:",(predconst!=pred0).sum().item(),"/",len(pred0))
print("Effective key-readout norm",(H0["k.weight"].T@qmean/math.sqrt(768)).norm().item())
lognote("Localization: after head LN, top ten dimensions carry 89.2% of decision and 90.0% of option energy. "
        f"Mean-query norm {qmean.norm().item():.4f}, query residual RMS norm "
        f"{(fit['Q']-qmean).square().sum(1).mean().sqrt().item():.4f}; "
        f"replacing query by mean changes {(predconst!=pred0).sum().item()}/{len(pred0)} fit predictions. "
        "Candidate: rank-one addition to k.weight along mean-query direction, reading option-state contrasts. "
        "No weights changed yet.")
lognote("## Before pass 5\nExact edit: none. READ original development 180 items/900 questions; IDs in read_dev.json. "
        "Development is not used to derive candidate directions, only to choose regularization and strength.")
```
