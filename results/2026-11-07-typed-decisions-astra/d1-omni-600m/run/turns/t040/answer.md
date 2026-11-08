I’ll inspect the remaining invoice failures and measure the conditional edit’s actual cue responses. If the remaining errors cluster around a specific missed cue, that may justify one more targeted repair; otherwise I’ll keep the validated version.
```python
# Hypothesis: remaining invoice failures reflect weak encoding of specific reconciliation cues rather than a general calibration problem.
cf14=torch.tensor(np.nan_to_num(np.load(WORK/"fwd_14/option_logits.npy"),nan=0))
cf14v=cf_values(cf14)
for family,low,high,q in cfcontrasts:
    ids=sorted({t["id"] for t in cf_tags if t["family"]==family})
    b=np.mean([cfvb[family,i,high,q]-cfvb[family,i,low,q] for i in ids])
    v1=np.mean([cfve[family,i,high,q]-cfve[family,i,low,q] for i in ids])
    v2=np.mean([cf14v[family,i,high,q]-cf14v[family,i,low,q] for i in ids])
    print("Actual cue response",family,q,round(b,4),"->",round(v1,4),"->",round(v2,4))
print("\nRemaining invoice failures on dev:")
invoice_bad=[]
for i in split_ids["invoice_processing"]["dev"]:
    r=ed12_by_id[i];p=probs_record(r);g=by_id[i]["labels"]["gold"]
    q="discrepancy_severity"
    pr=max(p[q],key=p[q].get);ta=max(g[q]["probabilities"],key=g[q]["probabilities"].get)
    if pr!=ta:invoice_bad.append((p[q][pr],i,pr,ta))
for prob,i,pr,ta in sorted(invoice_bad,reverse=True)[:5]:
    print(i,"pred",pr,round(prob,3),"gold",ta)
    print(json.dumps(by_id[i]["inputs"]["state"]))
    print("gold probabilities",by_id[i]["labels"]["gold"]["discrepancy_severity"]["probabilities"])
print("Actual conditional vs local replay error:",float((cf_center(cf14)-cf_center(logits_scorer_params(cfx,cond_params))).abs().max()))
note("Pass14 actual conditional counterfactuals measured; local scorer replay remains numerically accurate. Confirmed state-dependent responses, including discount and freight cues absent from simpler repair. Exploring remaining invoice errors only on dev; no test or hidden-label access. Current best remains best_2 (validation +14.20pp SE1.543pp).")
```
