Validation confirms the edit: accuracy rises from 58.2% to 72.87%, a 14.67-point gain with paired SE 1.55 points. I’ll back it up, then probe whether it responds appropriately when specific state cues change—such as delivery exceptions, payment deadlines, or constraint violations.

```python
# Hypothesis: the validated key direction restores sensitivity to concrete state cues rather than memorizing record-level labels.
import shutil, copy
lognote("Pass 10 hidden validation: original .582000 -> edited .728667, gain .146667, "
        "paired SE .015527; 163 improved, 43 worsened. Rank-one key edit is validation-confirmed.")
BEST1=WORK/"best_1"
BEST1.mkdir(exist_ok=True)
shutil.copytree(WORK/"edited",BEST1/"edited",dirs_exist_ok=True)
torch.save({"k.weight":load("k.weight"),"u":u,"v":v0},BEST1/"weights_and_direction.pt")
json.dump({"validation_gain":.14666666666666667,"validation_se":.015527393948731169,
           "test_gain":.13555555555555557,"test_se":.021871219240143364,
           "formula":"K0 + outer(u,v), ridge=.001, fit=540, strength=1"},
          open(BEST1/"results.json","w"))
print("Backed up confirmed edit:",BEST1)
counter_requests=[]
counter_meta=[]
def make_request(i,state):
    req=json.loads(recbyid[i]["calls"][0]["prompt"])
    req["state"]["inputs"]["state"]=state
    return req
fitids_ordered=sorted(splits["fit"],key=lambda i:int(i.split("_")[-1]))
invoice_ids=[i for i in fitids_ordered if byid[i]["inputs"]["workflow"]=="invoice_processing"]
agent_ids=[i for i in fitids_ordered if byid[i]["inputs"]["workflow"]=="agent_trace_observability"]
security_ids=[i for i in fitids_ordered if byid[i]["inputs"]["workflow"]=="security_incidents"]
def add_pair(i,cue,s0,s1,qid,positive,negative):
    start=len(counter_requests)
    counter_requests.extend([make_request(i,s0),make_request(i,s1)])
    counter_meta.append({"base_id":i,"cue":cue,"requests":[start,start+1],"qid":qid,
                         "positive":positive,"negative":negative})
for i in invoice_ids[:10]:
    s=copy.deepcopy(byid[i]["inputs"]["state"])
    s["invoice"]["lines"][0]["qty"]=s["purchase_order"]["lines"][0]["qty"]
    s["invoice"]["lines"][0]["unit_usd"]=s["purchase_order"]["lines"][0]["unit_usd"]
    s["invoice"]["lines"][0]["total_usd"]=s["purchase_order"]["total_usd"]
    s["invoice"]["total_usd"]=s["purchase_order"]["total_usd"]
    s["delivery"]["received_qty"]=s["purchase_order"]["lines"][0]["qty"]
    s["delivery"]["condition"]="accepted"
    a=copy.deepcopy(s); b=copy.deepcopy(s)
    b["delivery"]["condition"]="accepted with exceptions"
    add_pair(i,"delivery_exceptions",a,b,"matches_order","false","true")
    a=copy.deepcopy(s); b=copy.deepcopy(s)
    for ss in [a,b]:
        ss["payment"]["discount_expires_in_days"]=None
        ss["payment"]["early_payment_discount_pct"]=None
        ss["payment"]["status"]="scheduled"
    a["payment"]["days_until_due"]=20
    b["payment"]["days_until_due"]=1
    add_pair(i,"payment_deadline",a,b,"urgency","3","1")
for i in agent_ids[:10]:
    a=copy.deepcopy(byid[i]["inputs"]["state"]); b=copy.deepcopy(a)
    a["trace_summary"]["constraint_violations"]=0
    b["trace_summary"]["constraint_violations"]=2
    add_pair(i,"constraint_violation",a,b,"risk","3","0")
for i in security_ids[:10]:
    a=copy.deepcopy(byid[i]["inputs"]["state"]); b=copy.deepcopy(a)
    a["context"]["source_on_allowlist"]=False
    b["context"]["source_on_allowlist"]=True
    add_pair(i,"source_allowlisted",a,b,"true_positive","false","true")
json.dump({"type":"read","model":"original","requests":counter_requests},
          open(WORK/"counterfactual_original.json","w"))
json.dump({"type":"read","model":"edited","requests":counter_requests},
          open(WORK/"counterfactual_edited.json","w"))
json.dump(counter_meta,open(WORK/"counterfactual_pairs.json","w"))
print("Counterfactual requests:",len(counter_requests),"questions:",len(counter_requests)*5)
print("\nRemaining development errors after rank-one edit:")
pnew=(dev["L"]+Cdev).masked_fill(~dev["mask"],-1e9).argmax(1)
for g in ["agent_trace_observability/action","invoice_processing/discrepancy_severity","customer_service/action","security_incidents/urgency"]:
    ix=[i for i,x in enumerate(dev["group"]) if x==g and pnew[i]!=dev["gold"][i].argmax()]
    print(g,"errors",len(ix),"/45")
    if ix:
        j=ix[0]; m=dev["meta"][j]
        print(dev["ids"][j],"new",m["options"][pnew[j]],"gold",m["options"][dev["gold"][j].argmax()])
        print(json.dumps(byid[dev["ids"][j]]["inputs"]["state"],ensure_ascii=False)[:950])
lognote("Current best: best_1/edited, validated rank-one k.weight edit. "
        "Counterfactual plan: 10 paired examples each for delivery accepted/exceptions, due in 20/1 days, "
        "0/2 constraint violations, source allowlist false/true. All derive from fit items; no gold labels invented.")
lognote("## Before pass 11\nExact current edit: validated k.weight = K0 + outer(u,v), lambda .001; "
        "READ model=original ignores edit, on 80 synthetic single-cue counterfactual requests (400 questions), "
        "exact contents counterfactual_original.json. Pairs and expected margin signs in counterfactual_pairs.json.")
```
