The repair helps most questions, but customer actions and invoice discrepancies remain difficult. I’ll inspect a few fitting-set failures and measure which activation directions the edit uses. This will help determine whether the remaining errors justify another edit or are better treated as limitations.


```python
# Hypothesis: the successful edit reads low-energy state evidence, while remaining failures involve finer action and numerical distinctions.
print("Qwen head original/edit geometry:")
K0 = H0["k.weight"]
Kbest = load("k.weight")
dkbest = Kbest - K0
print("||K0||, ||deltaK||:", K0.norm().item(), dkbest.norm().item(),
      "cos(deltaK,K0):", torch.nn.functional.cosine_similarity(dkbest.flatten(), K0.flatten(), dim=0).item())
print("u unit norm:", u.norm().item(), "rank1 reconstruction max error:", (dkbest-torch.outer(u,v_ref)).abs().max().item())
print("Top 10 decision energy dims:", torch.argsort(fit["d"].square().mean(0),descending=True)[:10].tolist())
e_opt = fit["o"][fit["mask"]].square().mean(0)
top10 = torch.argsort(e_opt,descending=True)[:10]
print("Option top10 energy share:", (e_opt[top10].sum()/e_opt.sum()).item(),
      "v squared norm share on those dims:", (v_ref[top10].square().sum()/v_ref.square().sum()).item())
print("Top |v| coords:", [(int(j), round(v_ref[j].item(),3), round(e_opt[j].item(),3)) for j in v_ref.abs().argsort(descending=True)[:12]])

fit_pred_ref = (fit["L"]+Cfit_ref).masked_fill(~fit["mask"],-torch.inf).argmax(-1)
fit_tgt = fit["gold"].argmax(-1)
for gg in ["customer_service/action", "invoice_processing/discrepancy_severity", "security_incidents/urgency"]:
    idx = [j for j,g in enumerate(fit["group"]) if g == gg and fit_pred_ref[j] != fit_tgt[j]]
    print("\nRESIDUAL FAILURES", gg)
    for j in idx[:2]:
        eid = fit["ids"][j]
        ee = ex_by_id[eid]
        opts = fit["meta"][j]["options"]
        print(eid, "prediction", opts[fit_pred_ref[j]], "target", opts[fit_tgt[j]],
              "gold", ee["labels"]["gold"][gg.split("/")[1]]["probabilities"])
        print("state", json.dumps(ee["inputs"]["state"],ensure_ascii=False))
        print("question", json.dumps(ee["labels"]["questions"][gg.split("/")[1]],ensure_ascii=False))
```
