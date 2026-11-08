The pooled contrast is too close to a generic “prefer the unchosen option” signal. I’ll narrow the analysis to invoice urgency, where all 50 read examples chose “routine.” I’ll compare near-due and later-due invoices to see whether deadline information survives in the option representations, separately from the fixed option wording.
```python
# Hypothesis: invoice deadline evidence survives in option representations but is weak relative to a fixed routine-option preference.
def unit_center_features(x):
    c = x - np.nanmean(x,axis=-1,keepdims=True)
    return c/np.linalg.norm(c,axis=-1,keepdims=True)
TU2 = unit_center_features(T2)
invoice_rows = [r for r in rowinfo2 if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
near = np.array([by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"]<=3 for r in invoice_rows])
for name, feats in [("trunk",TU2),("head",U2)]:
    contrasts = np.stack([feats[r["i"],2]-feats[r["i"],1] for r in invoice_rows])
    shared = contrasts.mean(0)
    deadline = contrasts[near].mean(0)-contrasts[~near].mean(0)
    deadline -= shared*(deadline@shared)/(shared@shared)
    deadline /= np.linalg.norm(deadline)
    projections = contrasts@deadline
    a = contrasts[near][::2].mean(0)-contrasts[~near][::2].mean(0)
    b = contrasts[near][1::2].mean(0)-contrasts[~near][1::2].mean(0)
    print(name,"deadline double-diff norm",float(np.linalg.norm(contrasts[near].mean(0)-contrasts[~near].mean(0))),
          "split cosine",float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
          "near vs later projection",float(projections[near].mean()),float(projections[~near].mean()),
          "SD",float(projections.std()))
    if name=="head":
        d_deadline=deadline
# Isolate an invoice-urgency option direction from the universal scorer preference.
other_contrasts = np.stack([d for k,d in contrast_groups.items() if k!=("invoice_processing","urgency")])
universal = other_contrasts.mean(0)
basis = np.stack([eff_unit,universal/np.linalg.norm(universal),np.ones(1024)/32.0],axis=1)
Q,_ = np.linalg.qr(basis)
inv_contrast = contrast_groups[("invoice_processing","urgency")]
d_invoice = inv_contrast - Q@(Q.T@inv_contrast)
d_invoice /= np.linalg.norm(d_invoice)
print("Invoice-specific residual direction; norm before unit normalization",
      np.linalg.norm(inv_contrast-Q@(Q.T@inv_contrast)))
for key in sorted(groups):
    rr=[r for r in rowinfo2 if (r["wf"],r["q"])==key]
    gaps=np.array([(U2[r["i"],r["gi"]]-U2[r["i"],r["pi"]])@d_invoice for r in rr if not r["ok"]])
    print(key,round(float(gaps.mean()),4))
print("Invoice target-vs-routine directional mean near/later",
      [(flag, float(np.mean([(U2[r["i"],2]-U2[r["i"],1])@d_invoice for r,fl in zip(invoice_rows,near) if fl==flag]))) for flag in [False,True]])
read_req_b={"type":"read","model":"original","from":{"pass":1,"ids":read_ids_b},"layers":[14,16]}
(WORK/"read_b.json").write_text(json.dumps(read_req_b))
note("## Pass 3 planned: discovery activations B\nExact edits: none. Read remaining 200 discovery cases (50/workflow; read_b.json), all 1000 questions. Purpose: test replication of invoice deadline double-difference and invoice-specific option contrast on independent discovery half. Pooled error direction rejected as too close to generic anti-confidence. No edits computed from confirmation.")
```
