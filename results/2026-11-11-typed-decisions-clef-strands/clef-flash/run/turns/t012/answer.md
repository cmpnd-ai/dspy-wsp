The read batches may be returned in recorded-file order rather than requested-ID order. I’ll verify every row against the baseline probabilities, then build a fit-only correction using differences between options within each question. That targets ranking errors, not a global temperature.


```python
# Hypothesis: a centered readout correction can repair option ranking while preserving the fixed state and option representations.
def center_options(t, m):
    t = torch.nan_to_num(t)
    mm = m.to(t.dtype)
    if t.ndim==3:
        mm = mm.unsqueeze(-1)
    return (t-(t*mm).sum(1,keepdim=True)/mm.sum(1,keepdim=True).clamp_min(1))*mm

def read_batch(k, requested_ids):
    dd = WORK/f"fwd_{k}"
    meta = json.loads((dd/"meta.json").read_text())
    a = {p.stem:torch.from_numpy(np.array(np.load(p))) for p in dd.glob("*.npy")}
    wanted = set(requested_ids)
    ordered = [str(r["id"]) for r in base if str(r["id"]) in wanted]
    mismatch, reqmap = [], {}
    for i,m in enumerate(meta["rows"]):
        exid = ordered[m["request"]]
        probs = json.loads(basebyid[exid]["calls"][0]["completion"])[m["question"]]["probabilities"]
        expect = torch.tensor([probs[o] for o in m["options"]])
        mismatch.append((a["probabilities"][i,:len(expect)]-expect).abs().max().item())
        reqmap[m["request"]] = exid
    print("Read",k,"ordering verified max probability error",max(mismatch))
    assert max(mismatch)<1e-5
    a["head_inputs"] = torch.nan_to_num(a["head_inputs"])
    with torch.no_grad():
        a["activations"] = F.gelu(F.linear(a["head_inputs"],W0,b0))
    a["meta"] = meta["rows"]
    a["ids"] = [reqmap[m["request"]] for m in meta["rows"]]
    a["gold"] = torch.zeros_like(a["option_logits"])
    a["target"] = torch.zeros(len(meta["rows"]),dtype=torch.long)
    for i,m in enumerate(meta["rows"]):
        g = byid[a["ids"][i]]["labels"]["gold"][m["question"]]["probabilities"]
        a["gold"][i,:len(m["options"])] = torch.tensor([g[o] for o in m["options"]])
        a["target"][i] = a["gold"][i].argmax()
    return a

batches = [read_batch(k,ids) for k,ids in zip([2,3,4],read_chunks)]
D = {}
for key in ["option_mask","option_logits","prior","cosine","probabilities","activations","gold","target"]:
    D[key] = torch.cat([a[key] for a in batches],0)
D["meta"] = sum([a["meta"] for a in batches],[])
D["ids"] = sum([a["ids"] for a in batches],[])
D["isfit"] = torch.tensor([i in set(splits["fit"]) for i in D["ids"]])
D["istune"] = ~D["isfit"]
D["mask"] = D["option_mask"].bool()
D["logits"] = torch.nan_to_num(D["option_logits"])
D["C"] = center_options(D["activations"],D["mask"])*gate
components = {
    "prior": D["prior"],
    "cosine": gate*jscale*D["cosine"],
    "residual": gate*(D["activations"]@wout+bout)
}
for k,t in components.items():
    print(k,"centered RMS",round(center_options(t,D["mask"])[D["mask"]].square().mean().sqrt().item(),4))

def summary_logits(logits, subset, verbose=False):
    pred = logits.masked_fill(~D["mask"],-1e9).argmax(-1)
    ok = (pred==D["target"]).float()
    baseok = (D["logits"].masked_fill(~D["mask"],-1e9).argmax(-1)==D["target"]).float()
    ix = torch.where(subset)[0]
    deltas = collections.defaultdict(list)
    for i in ix.tolist():
        deltas[D["ids"][i]].append((ok[i]-baseok[i]).item())
    values = np.array([np.mean(v) for v in deltas.values()])
    result = dict(accuracy=ok[subset].mean().item(),gain=values.mean(),
                  se=values.std(ddof=1)/np.sqrt(len(values)),n=len(values))
    if verbose:
        bywf=collections.defaultdict(list)
        for i in ix.tolist():
            bywf[byid[D["ids"][i]]["inputs"]["workflow"]].append((baseok[i].item(),ok[i].item()))
        for wf,v in sorted(bywf.items()):
            print(wf,np.array(v).mean(0).round(4))
    return result

print("Fit baseline:",summary_logits(D["logits"],D["isfit"]))
print("Tune baseline:",summary_logits(D["logits"],D["istune"]))
note("Activation alignment verified against all recorded probabilities; READ 'from' preserves generation-file order, not the ID list order. Padding sanitized before centering. Residual scorer features and activations now aligned with 500 fit/100 tune labels.")
```
