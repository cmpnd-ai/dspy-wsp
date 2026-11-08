I’ll test whether the proposed correction responds to the cues it is meant to recover. These counterfactual records change one field at a time—violations, tool errors, payment timing, freight terms, or security privileges—while holding the rest fixed.
```python
# Hypothesis: the residual readout should increase harmful/risky judgments for violations and urgency for overdue or expiring payments, not just shift workflow priors.
import copy
cf_requests=[]; cf_tags=[]
def add_cf(i,family,value,mutator):
    req=json.loads(base_by_id[i]["calls"][0]["prompt"])
    mutator(req["state"]["inputs"]["state"])
    cf_requests.append(req); cf_tags.append({"id":i,"family":family,"value":value})
for i in split_ids["agent_trace_observability"]["fit"][:8]:
    for v in [0,1,3]:
        add_cf(i,"constraint_violations",v,lambda s,v=v:s["trace_summary"].update(constraint_violations=v))
    for v in [0,1,4]:
        add_cf(i,"tool_errors",v,lambda s,v=v:s["trace_summary"].update(tool_errors=v,constraint_violations=0))
for i in split_ids["invoice_processing"]["fit"][:8]:
    for v in [30,7,1,-5]:
        add_cf(i,"days_until_due",v,lambda s,v=v:s["payment"].update(days_until_due=v,status="overdue" if v<0 else "scheduled",discount_expires_in_days=None,early_payment_discount_pct=None))
    for v in [None,2]:
        add_cf(i,"discount",v,lambda s,v=v:s["payment"].update(days_until_due=30,status="scheduled",discount_expires_in_days=v,early_payment_discount_pct=2.0 if v else None))
    for v in ["prepaid","billable"]:
        add_cf(i,"freight_terms",v,lambda s,v=v:s["purchase_order"].update(freight_terms="freight prepaid by vendor, not separately billable" if v=="prepaid" else "freight is separately billable to purchaser"))
for i in split_ids["security_incidents"]["fit"][:8]:
    for v in ["read","production"]:
        add_cf(i,"privileges",v,lambda s,v=v:s["principal"].update(privileges=["repo:read"] if v=="read" else ["deploy:production","secrets:read"],type="employee" if v=="read" else "admin"))
    for v in ["low","critical"]:
        add_cf(i,"asset_criticality",v,lambda s,v=v:s["context"].update(asset_criticality=v))
json.dump(cf_tags,open(WORK/"counterfactual_tags.json","w"))
json.dump({"type":"read","model":"original","requests":cf_requests},open(WORK/"counterfactual_original.json","w"))
candidate_dx=mode_solutions[(0.01,0.25)].clone()
pair_units=[int(j) for j in neuron_order[:2]]
mu_x=D["x"][fitmask].mean(0)
pair_removed=D["z"][:,:,pair_units]@W0.squeeze()[pair_units]
print("Counterfactuals:",len(cf_requests),len(cf_requests)*5,"questions")
print("Chosen direction: rel ridge .01, .25 mode + .75 soft target, norm",float(candidate_dx.norm()))
print("Chosen units",pair_units,"max removed option-contrast logit",float(centered(pair_removed).abs().max()))
for strength in [0,0.25,0.5,0.75,1,1.25,1.5]:
    l=D["logits"]-pair_removed+strength*(D["x"]@candidate_dx)
    print("Predicted dose",strength,{k:round(v,5) for k,v in evaluate_logits(l).items()})
note("Candidate frozen for causal tests: normalized-input residual, ridge relative=.01, target 5*T*(.75 gold probability + .25 gold mode), subtract original logits; fit on 448 examples. Delta norm "+str(float(candidate_dx.norm()))+
     ". Exact implementation planned: +/- GELU pair using scorer neurons 33 and 746; original contrast contribution <1e-6. No edits yet.\n"
     "Next pass: READ original 144 controlled requests (720 questions), exact requests/tags in counterfactual_original.json and counterfactual_tags.json. Preregistered effects: violations raise harmful/risk; errors reduce success; overdue/discount raises urgency; production privileges raise security severity.")
```
