I’ll build matched requests that vary topic, duplicate status, invoice amounts, security evidence, or cancellation intent while holding the rest of each record fixed. These are diagnostic probes, not fitting data. I’ll also repeat the earlier numeric probes to check the repair’s limits.
```python
# Hypothesis: the frozen head restores some semantic cue responses but does not necessarily repair numeric comparisons or exact ID matching.
import copy
diagnostic_requests = copy.deepcopy(cue_requests)
diagnostic_manifest = copy.deepcopy(cue_info)
def append_diagnostic(base_id, kind, value, state, questions):
    original_req = json.loads(baseline_by_id[base_id]["calls"][0]["prompt"])
    original_req["state"]["inputs"]["state"] = copy.deepcopy(state)
    original_req["questions"] = {q:original_req["questions"][q] for q in questions}
    diagnostic_requests.append(original_req)
    diagnostic_manifest.append({"base_id":base_id,"kind":kind,"value":value})
bases_diag = {wf:[ex["id"] for ex in examples if ex["inputs"]["workflow"]==wf][:6]
              for wf in sorted(set(wf900))}
for base_id in bases_diag["invoice_processing"]:
    state = copy.deepcopy(by_id[base_id]["inputs"]["state"])
    iid = state["invoice"]["id"]
    previous = [x for x in state["vendor_history"]["prior_invoice_ids"] if x != iid]
    if len(previous) < 3:
        previous += ["INV-2025-0041"]*(3-len(previous))
    for duplicated in [False,True]:
        s = copy.deepcopy(state)
        s["vendor_history"]["prior_invoice_ids"] = previous[:-1]+([iid] if duplicated else [previous[-1]])
        append_diagnostic(base_id,"duplicate_id",int(duplicated),s,["duplicate","disposition"])
    for ratio in [1.0,1.1,1.5]:
        s = copy.deepcopy(state)
        poline = s["purchase_order"]["lines"][0]
        qty, unit = poline["qty"], poline["unit_usd"]
        s["delivery"]["received_qty"] = qty
        s["delivery"]["condition"] = "accepted"
        s["purchase_order"]["total_usd"] = round(qty*unit,2)
        s["invoice"]["lines"] = [{"qty":qty,"sku":"SKU-940",
                                  "total_usd":round(qty*unit*ratio,2),
                                  "unit_usd":round(unit*ratio,2)}]
        s["invoice"]["total_usd"] = round(qty*unit*ratio,2)
        s["vendor_history"]["prior_invoice_ids"] = previous
        append_diagnostic(base_id,"invoice_amount_ratio",ratio,s,
                          ["matches_order","discrepancy_severity","disposition"])
security_evidence = {
    "authorized":"The account owner confirmed this was their own login from a managed company device. The session matches an approved change ticket. The authentication and subsequent activity were authorized.",
    "compromised":"The account owner confirmed they did not initiate this login. A stolen password was used from a known attacker address, and the resulting session accessed restricted resources without authorization."
}
for base_id in bases_diag["security_incidents"]:
    state = copy.deepcopy(by_id[base_id]["inputs"]["state"])
    state["alert"]["rule"] = "suspicious_login"
    state["alert"]["description"] = "A successful authentication event was detected."
    for value,text in security_evidence.items():
        s = copy.deepcopy(state)
        s["alert"]["evidence"] = text
        append_diagnostic(base_id,"authorization_evidence",value,s,
                          ["credential_compromise","true_positive","disposition"])
churn_text = {
    "satisfied":"Thanks, that answers my question. I am happy with the service and plan to keep my subscription.",
    "frustrated":"This problem still is not fixed. I have contacted you several times and am getting frustrated.",
    "leaving":"This problem still is not fixed. I have decided to cancel my subscription and move to another provider. Please close my account."
}
for base_id in bases_diag["customer_service"]:
    state = copy.deepcopy(by_id[base_id]["inputs"]["state"])
    for value,text in churn_text.items():
        s = copy.deepcopy(state)
        s["thread"] = [{"role":"customer","text":text}]
        append_diagnostic(base_id,"cancellation_intent",value,s,["churn_risk","needs_human","action"])
num_diag_questions = sum(len(r["questions"]) for r in diagnostic_requests)
assert num_diag_questions <= 1024
(work/"final_cue_manifest.json").write_text(json.dumps(diagnostic_manifest,indent=2))
for mode in ["original","edited"]:
    req = {"type":"read","model":mode,"requests":diagnostic_requests}
    (work/f"final_cue_{mode}_request.json").write_text(json.dumps(req))
print("Matched diagnostic requests/questions:",len(diagnostic_requests),num_diag_questions)
print("Kinds:",dict(__import__("collections").Counter(m["kind"] for m in diagnostic_manifest)))
assert torch.equal(load("scalar.weight"),frozen_weight)
note("\nPlanned forward22: original READ matched synthetic requests, final_cue_original_request.json, "+
     str(len(diagnostic_requests))+" requests/"+str(num_diag_questions)+" questions. Manifest final_cue_manifest.json. Current edit restored frozen best4 scalar.weight=w0+.875*d_mix900; original mode ignores it. Cue tests include previous numeric/topic probes plus duplicate-ID, amount mismatch, authorization evidence and cancellation-intent variants. Diagnostic only, no new fitting.")
```
