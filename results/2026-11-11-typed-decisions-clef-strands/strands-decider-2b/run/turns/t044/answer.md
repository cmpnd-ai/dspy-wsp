Almost none of the edit’s weight lies on the ten dimensions that dominate activation energy. I’ll split the repair into contributions from those dimensions and the remaining dimensions, then test them separately. This distinguishes an outlier adjustment from recovery of weaker state evidence.


```python
# Hypothesis: most of the gain comes from reading low-energy option-state dimensions rather than modifying dominant activation outliers.
def eval_cached(dd, logits, reference=None):
    mask = dd["mask"]
    pp = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    ok = (pp.argmax(-1) == dd["gold"].argmax(-1)).float()
    ref = dd["L"] if reference is None else reference
    ok0 = (ref.masked_fill(~mask,-torch.inf).argmax(-1) == dd["gold"].argmax(-1)).float()
    eid_order = list(dict.fromkeys(dd["ids"]))
    item_scores = []
    item_delta = []
    for eid in eid_order:
        idx = [i for i, v in enumerate(dd["ids"]) if v == eid]
        item_scores.append(ok[idx].mean().item())
        item_delta.append((ok[idx]-ok0[idx]).mean().item())
    item_scores, item_delta = np.array(item_scores), np.array(item_delta)
    return {"accuracy": float(item_scores.mean()), "gain": float(item_delta.mean()),
            "se": float(item_delta.std(ddof=1)/np.sqrt(len(item_delta))),
            "brier": float(((pp-dd["gold"]).square().sum(-1)).mean())}

v_outlier = torch.zeros_like(v_ref)
v_outlier[top10] = v_ref[top10]
v_lowenergy = v_ref - v_outlier
dev_out = correction(dev, v_outlier)
dev_low = correction(dev, v_lowenergy)
print("Top10 dims:", top10.tolist())
for name, vv_ablate in [("none", torch.zeros_like(v_ref)),("top10_only",v_outlier),
                        ("other2038_only",v_lowenergy),("full_best2",v_ref)]:
    print(name, eval_cached(dev, dev["L"] + correction(dev, vv_ablate)))
for name, cc in [("top10",dev_out),("other2038",dev_low),("full",Cdev_ref)]:
    print(name, "RMS centered logit change", float(cc[dev["mask"]].square().mean().sqrt()))

# Fit-only option-schema span: a read direction orthogonal to this cannot change fit mean schema preferences.
schema_xmeans = []
for gg in sorted(set(fit["group"])):
    idx = torch.tensor([i for i,g in enumerate(fit["group"]) if g==gg])
    nopt = int(fit["mask"][idx[0]].sum())
    schema_xmeans.append(Xfit3[idx, :nopt].mean(0))
MU = torch.cat(schema_xmeans,0).double()
_, mu_svals, mu_Vt = torch.linalg.svd(MU, full_matrices=False)
mu_rank = int((mu_svals > mu_svals.max()*1e-5).sum())
Pmu = mu_Vt[:mu_rank].float().T
v_mean_span = Pmu @ (Pmu.T @ v_ref)
v_schema_null = v_ref-v_mean_span
print("Schema mean span rank",mu_rank)
for name,vv_ablate in [("schema_mean_span_only",v_mean_span),("schema_mean_null_only",v_schema_null)]:
    print(name,eval_cached(dev,dev["L"]+correction(dev,vv_ablate)),
          "direction norm share",float(vv_ablate.norm()/v_ref.norm()))

with open(WORK/"notes.md","a") as f:
    f.write("\nGeometry of best2: ||K0||=24.6776, ||deltaK||=261.2993; "
            "cos(deltaK,K0)=-.000954, rank1 reconstruction error <2.4e-7. "
            "Ten normalized option dimensions carry 89.987% of activation energy but only .00434% of v squared norm. "
            "Large relative parameter change compensates for low feature amplitude, not global scaling.\n")
    for name,vv_ablate in [("top10_only",v_outlier),("other2038_only",v_lowenergy),
                          ("schema_mean_span_only",v_mean_span),("schema_mean_null_only",v_schema_null)]:
        f.write(f"Best2 cached dev ablation {name}: {eval_cached(dev,dev['L']+correction(dev,vv_ablate))}\n")
```
