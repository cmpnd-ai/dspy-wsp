Correct-minus-chosen contrasts are reproducible, but their pooled direction points mostly against the existing readout—as expected when selecting errors. I’ll remove that component before considering an edit, so the intervention cannot merely flatten or reverse the model’s preferences. I’ll also check whether the remaining direction favors the intended options across questions.
```python
# Hypothesis: a residual error direction orthogonal to the current readout captures a specific option-selection mismatch, not global confidence.
eff_unit = effective / np.linalg.norm(effective)
residual2 = mean_diff2 - (mean_diff2 @ eff_unit)*eff_unit
residual2 -= residual2.mean()
d_residual = residual2 / np.linalg.norm(residual2)
print("Pooled residual norm:",float(np.linalg.norm(residual2)),"readout cosine:",float(d_residual@eff_unit))
print("Raw logit margin stats on errors:",
      np.quantile([arr2["option_logits"][r["i"],r["pi"]]-arr2["option_logits"][r["i"],r["gi"]] for r in all_errors2],[.1,.5,.9]))
for key in sorted(groups):
    rr = [r for r in rowinfo2 if (r["wf"],r["q"])==key]
    delta = np.array([(U2[r["i"],r["gi"]]-U2[r["i"],r["pi"]])@d_residual for r in rr if not r["ok"]])
    ranks = [int(np.nanargmax(U2[r["i"],:len(r["opts"])]@d_residual))==r["gi"] for r in rr]
    c = contrast_groups[key]
    cr = c-(c@eff_unit)*eff_unit
    print(key,"err target-minus-chosen",round(float(delta.mean()),4),
          "direction alone gold rank",round(float(np.mean(ranks)),2),
          "readout cosine",round(float(c@eff_unit/np.linalg.norm(c)),3),
          "residual alignment",round(float(cr@d_residual/np.linalg.norm(cr)),3))
# Record state-specific patterns without using confirmation records.
invurg = [r for r in rowinfo2 if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
for isdue in [False,True]:
    rs = [r for r in invurg if (by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"] <= 3)==isdue]
    print("Invoice days_until_due<=3",isdue,"n",len(rs),
          "gold",collections.Counter(r["opts"][r["gi"]] for r in rs),
          "pred",collections.Counter(r["opts"][r["pi"]] for r in rs),
          "resid (2-1)",float(np.mean([(U2[r["i"],2]-U2[r["i"],1])@d_residual for r in rs])))
print("Top residual coordinates:",[(int(j),round(float(d_residual[j]),4)) for j in np.argsort(np.abs(d_residual))[-12:][::-1]])
```
