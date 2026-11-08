I’ll check the gain-2 results by question and verify that the head features are unchanged. A useful repair should move urgency logits in the predicted direction without relying on changes elsewhere in the network.
The analysis will compare paired accuracy, deadline-specific logit changes, and collateral effects. I’ll also save the gain-2 weights as a reproducible checkpoint.
```python
# Hypothesis: gain-2 payment readout changes deadline-dependent logits while preserving the head's feature computation.
arr7,rowinfo7,meta7=load_read(7,read_ids_b)
read_summary(arr7,rowinfo7,arr3,rowinfo3)
deadline_summary(arr7,rowinfo7)
print("Head feature max difference:",float(np.nanmax(np.abs(arr7["head_inputs"]-arr3["head_inputs"]))))
rs=[r for r in rowinfo7 if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
for r in rs[:8]:
    i=r["i"]
    pay=by_id[r["id"]]["inputs"]["state"]["payment"]
    print(r["id"],"payment",pay["days_until_due"],pay["discount_expires_in_days"],
          "gold",r["gi"],"old/new",rowinfo3[i]["pi"],r["pi"],
          "logit delta",np.round(arr7["option_logits"][i,:4]-arr3["option_logits"][i,:4],2))
def backup_edits(name):
    p=WORK/name
    p.mkdir(exist_ok=True)
    for n in edited():
        torch.save(load(n),p/(n+".pt"))
    (p/"manifest.json").write_text(json.dumps(edit_manifest(),indent=2))
    return p
backup_edits("payment_gain2")
note("Pass 7 results analyzed below in session; gain-2 checkpoint saved to payment_gain2. Upstream feature equality checked.")
```
