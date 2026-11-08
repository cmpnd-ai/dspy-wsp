The prior-only ablation scores 48.0%, versus 58.5% for the full edit on the same records. I’ll restore the validated edit and build controlled cue tests: changing only violations, customer topic, payment deadline, or asset criticality should change the corresponding decision in the expected direction.
```python
# Hypothesis: the repaired readout responds to decisive state cues, not merely different question/option priors.
note("Forward9 prior-only dev150:0.480000 vs original0.373333, gain0.106667 SE0.022118. Full direction0.585333 on same records. Record-specific component contributes beyond template correction.")
def read_records(k,model):
    return [json.loads(s) for s in (WORK/f"gen_{k}_{model}.jsonl").read_text().splitlines()]
r_full={r["id"]:r for r in read_records(6,"edited")}
r_prior={r["id"]:r for r in read_records(9,"edited")}
inc=np.array([r_full[i]["score"]-r_prior[i]["score"] for i in r_full])
print("Full minus prior-only:",inc.mean(),"paired SE",inc.std(ddof=1)/np.sqrt(len(inc)))
set_scalar_direction(d_soft,1.0)
cue_requests=[]; cue_info=[]
def add_cue(pid,kind,value,question_ids,modify):
    req=json.loads(baseline_by_id[pid]["calls"][0]["prompt"])
    modify(req["state"]["inputs"]["state"])
    req["questions"]={q:req["questions"][q] for q in question_ids}
    cue_requests.append(req);cue_info.append({"base_id":pid,"kind":kind,"value":value})
dev_by_wf={wf:[pid for pid in splits["dev"] if by_id[pid]["inputs"]["workflow"]==wf][:4] for wf in sorted({e["inputs"]["workflow"] for e in examples})}
for pid in dev_by_wf["agent_trace_observability"]:
    for v in [0,1,3]:
        def modify(s,v=v):
            s["trace_summary"]["constraint_violations"]=v
            s["trace_summary"]["tool_errors"]=0
            s["trace_summary"]["irreversible_actions"]=0
        add_cue(pid,"violations",v,["action","needs_review","outcome","risk"],modify)
topic_text={
    "account":"I forgot my account password and cannot sign in. Please help me reset my password and regain access to my account.",
    "billing":"There is an unexpected extra charge on this month's subscription bill. Please explain the charge on my billing statement.",
    "delivery":"My physical order has not arrived. The tracking link has not updated. Please tell me where the shipment is and when it will be delivered.",
    "refund":"I returned my purchase and was promised a refund. Please confirm when the refund will arrive in my bank account.",
    "technical":"The application crashes whenever I open a project. I get an error code and cannot use the software. Please help troubleshoot this technical problem.",
}
for pid in dev_by_wf["customer_service"]:
    for topic,text in topic_text.items():
        def modify(s,text=text):
            s["thread"]=[{"role":"customer","text":text}]
            s.pop("orders",None)
        add_cue(pid,"topic",topic,["category"],modify)
for pid in dev_by_wf["invoice_processing"]:
    for days in [30,7,0,-5]:
        def modify(s,days=days):
            s["payment"]["days_until_due"]=days
            s["payment"]["discount_expires_in_days"]=None
            s["payment"]["early_payment_discount_pct"]=None
            s["payment"]["status"]="overdue" if days<0 else "scheduled"
        add_cue(pid,"days_due",days,["urgency"],modify)
for pid in dev_by_wf["security_incidents"]:
    for critical in ["low","medium","high","critical"]:
        def modify(s,critical=critical): s["context"]["asset_criticality"]=critical
        add_cue(pid,"asset_criticality",critical,["severity"],modify)
(WORK/"cue_manifest.json").write_text(json.dumps(cue_info))
(WORK/"cue_original_request.json").write_text(json.dumps({"type":"read","model":"original","requests":cue_requests}))
(WORK/"cue_edited_request.json").write_text(json.dumps({"type":"read","model":"edited","requests":cue_requests}))
print("Counterfactual requests",len(cue_requests),"questions",sum(len(r["questions"]) for r in cue_requests))
note("Planned forward10: original READ controlled cue requests, exact cue_original_request.json (64 requests,100 questions; manifest cue_manifest.json). Current edit restored scalar.weight=w_original+d_soft. Synthetic variations are diagnostic only, not fitting labels: violations0/1/3; five customer topics; days_due30/7/0/-5; asset criticality low/medium/high/critical. Prediction: edited head has stronger correctly directed responses, especially agent violations and customer category.")
```
