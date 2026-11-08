I’ll probe invoice discrepancies directly: hold each invoice’s context fixed and vary price, billed quantity, delivery quantity, or delivery condition. This will show whether the remaining problem is weak arithmetic evidence, weak condition-word evidence, or both.
```python
# Hypothesis: the scorer under-reads invoice mismatch and delivery-exception cues, with different sensitivity to numeric and textual evidence.
import copy
inv_probe_requests, inv_probe_tags = [], []
inv_fit_ids = [i for i in splits["fit"] if by_id[i]["inputs"]["workflow"]=="invoice_processing"][:8]
print("Probe sources:", inv_fit_ids)
print("Request format:", type(cf_requests[0]["state"]), list(cf_requests[0]["questions"]))
for iid in inv_fit_ids:
    st = copy.deepcopy(by_id[iid]["inputs"]["state"])
    qty = st["purchase_order"]["lines"][0]["qty"]
    unit = st["purchase_order"]["lines"][0]["unit_usd"]
    st["purchase_order"]["total_usd"] = round(qty*unit,2)
    st["delivery"]["received_qty"] = qty
    st["delivery"]["condition"] = "accepted"
    st["invoice"]["lines"] = [{"qty":qty, "sku":"SKU-215", "unit_usd":unit, "total_usd":round(qty*unit,2)}]
    st["invoice"]["total_usd"] = round(qty*unit,2)
    st["invoice"]["id"] = "INV-2026-NEW"
    st["payment"].update({"days_until_due":14, "early_payment_discount_pct":None, "discount_expires_in_days":None, "status":"scheduled"})
    variations = [("matched",0), ("price",1.01),("price",1.10),("price",1.40),
                  ("billed_qty",1.10),("billed_qty",1.40),("received_qty",0.9),("received_qty",0.6),
                  ("condition","accepted with exceptions"),("condition","damaged"),("condition","rejected")]
    req_src = next(req for req, tag in zip(cf_requests, cf_tags) if "invoice" in str(req["questions"]) or tag.get("workflow")=="invoice_processing") if False else None
    calls = base_by_id[iid]["calls"] if "calls" in base_by_id[iid] else base_by_id[iid].get("model_calls")
    if calls is None:
        print("Record keys:",base_by_id[iid].keys())
        break
    request = json.loads(calls[0]["prompt"])
    for family,val in variations:
        ss=copy.deepcopy(st)
        line=ss["invoice"]["lines"][0]
        if family=="price":
            line["unit_usd"]=round(unit*val,2)
        elif family=="billed_qty":
            line["qty"]=max(qty+1, round(qty*val))
        elif family=="received_qty":
            ss["delivery"]["received_qty"]=round(qty*val)
        elif family=="condition":
            ss["delivery"]["condition"]=val
        line["total_usd"]=round(line["qty"]*line["unit_usd"],2)
        ss["invoice"]["total_usd"]=line["total_usd"]
        inv_probe_requests.append({"state":ss,"questions":copy.deepcopy(request["questions"])})
        inv_probe_tags.append({"id":iid,"family":family,"value":val})
if inv_probe_requests:
    json.dump({"type":"read","model":"original","requests":inv_probe_requests},open(W/"invoice_cues_original.json","w"))
    json.dump(inv_probe_tags,open(W/"invoice_cue_tags.json","w"))
    note(f"\nNext pass15: READ original {len(inv_probe_requests)} controlled invoice requests ({5*len(inv_probe_requests)} questions) in invoice_cues_original.json. Sources {inv_fit_ids}; no probe labels fitted. Installed edits remain validated best_2: conditional_candidate.pt (scorer.1 rows/biases pair33/746 plus108 gated neurons; scorer.3 corresponding110 entries). Original model requested for cue baseline.")
    print("Prepared",len(inv_probe_requests),"requests.")
```
The probe requests use only fit-set records as templates; they are for measuring cue sensitivity, not for fitting new weights.
