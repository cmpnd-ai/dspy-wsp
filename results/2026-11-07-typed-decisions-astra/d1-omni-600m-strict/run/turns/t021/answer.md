The strong MLP attenuation gained only 0.6 percentage points and did not fix invoice urgency, so I’ve restored it. I’ll now isolate the deadline cue with matched counterfactuals: identical invoices with only the due date, payment status, and discount deadline changed. This avoids confusing deadline information with other differences between invoices.
```python
# Hypothesis: the head encodes payment-deadline changes, but the scorer fails to turn that causal signal into urgency rankings.
import copy
synthetic_requests=[]
synthetic_tags=[]
invoice_seed_ids=[eid for eid in read_ids_a if by_id[eid]["inputs"]["workflow"]=="invoice_processing"][:16]
variants=[
    ("later",{"days_until_due":30,"status":"scheduled","discount_expires_in_days":None,"early_payment_discount_pct":None}),
    ("near",{"days_until_due":1,"status":"pending","discount_expires_in_days":None,"early_payment_discount_pct":None}),
    ("overdue",{"days_until_due":-10,"status":"overdue","discount_expires_in_days":None,"early_payment_discount_pct":None}),
    ("discount",{"days_until_due":30,"status":"scheduled","discount_expires_in_days":1,"early_payment_discount_pct":2.0})
]
for eid in invoice_seed_ids:
    base_req=json.loads(base_by_id[eid]["calls"][0]["prompt"])
    for tag,changes in variants:
        req=copy.deepcopy(base_req)
        req["state"]["inputs"]["state"]["payment"].update(changes)
        synthetic_requests.append(req)
        synthetic_tags.append({"id":eid,"variant":tag})
syn_req={"type":"read","model":"original","requests":synthetic_requests,"layers":[16]}
(WORK/"deadline_counterfactuals.json").write_text(json.dumps(syn_req))
(WORK/"deadline_counterfactual_tags.json").write_text(json.dumps(synthetic_tags))
note("Pass 5: .85 mode attenuation discovery A +.0060 +/- .00280 paired SE, but invoice urgency unchanged; logit corr .9907, scale .8194. Restored; not kept.\n"
     "## Pass 6 planned: matched payment-deadline counterfactuals\nExact edits: none. Original reads 64 synthetic requests (16 discovery-A invoices x later/near/overdue/discount), all 320 questions. Only payment days/status/discount fields change; deadline_counterfactuals.json and tags record exact cases. Goal: isolate causal deadline representation before targeted scorer edit.")
print("Synthetic requests",len(synthetic_requests),"edits",edited())
print("Small-output scorer neurons",[(int(j),float(w3[j])) for j in np.argsort(np.abs(w3))[:8]])
```
