I’ll build controlled probes for the extension’s three target fields: agent urgency, customer action, and customer need for a human. Each probe changes a specific state cue while keeping the schema fixed. Comparing the original, shared repair, and full repair will isolate what the gated neurons actually add.


```python
# Hypothesis: the targeted extension selectively strengthens urgency and authority/resolution cues without changing unrelated question readouts.
import copy
target_probe_requests=[]
target_probe_meta=[]
agent_seeds=[ii for ii in splits["fit"] if byid[ii]["inputs"]["workflow"]=="agent_trace_observability"][:12]
customer_seeds=[ii for ii in splits["fit"] if byid[ii]["inputs"]["workflow"]=="customer_service"][:12]
def record_probe(seed,family,level,request):
    target_probe_requests.append(request)
    target_probe_meta.append({"seed":seed,"family":family,"level":level})
for ii in agent_seeds:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for family,levels in [("urgency_constraint_violations",[0,1,2,4]),("urgency_irreversible_actions",[0,1,2,4])]:
        for level in levels:
            rq=copy.deepcopy(rq0)
            ss=rq["state"]["inputs"]["state"]["trace_summary"]
            ss.update({"constraint_violations":0,"irreversible_actions":0,"tool_errors":0})
            ss["constraint_violations" if "constraint" in family else "irreversible_actions"]=level
            record_probe(ii,family,level,rq)
customer_texts={
    "routine_information":"Could you explain how to update the email address on my account?",
    "human_authority":"The automated assistant has failed three times to update my email address. Please transfer me to a human support agent who has authority to fix this.",
    "resolved":"A human support agent fixed the issue, and I verified my account email is correct. Everything is resolved. Thank you; no further action is needed.",
    "verified_refund":"I was billed twice for one order. Both charges have settled, and the duplicate charge has been verified. Please issue the refund that is due.",
    "missing_information":"I have a problem, but I haven't told you what happened, which order it concerns, or what I want. Can you help?"
}
for ii in customer_seeds:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for level,text in customer_texts.items():
        rq=copy.deepcopy(rq0)
        rq["state"]["inputs"]["state"]["thread"]=[{"role":"customer","text":text}]
        record_probe(ii,"customer_resolution_authority",level,rq)
for model in ["original","edited"]:
    json.dump({"type":"read","model":model,"requests":target_probe_requests},open(WORK/f"target_cue_probes_{model}.json","w"))
json.dump(target_probe_meta,open(WORK/"target_cue_probe_meta.json","w"))
print("Controlled requests:",len(target_probe_requests),"questions",sum(len(r["questions"]) for r in target_probe_requests))
with open(WORK/"notes.md","a") as f:
    f.write("\n## Planned pass 16 — original target-cue probes\nCurrent edit best_2 unchanged. READ original on 156 controlled requests, 780 questions: 12 fit agent seeds x (4 violation counts +4 irreversible-action counts); 12 fit customer accounts x5 routine/authority/resolved/refund/underspecified threads. Files target_cue_probes_original.json and target_cue_probe_meta.json. No gold/score. Prespecified expectations: urgency critical-routine increases with violations/irreversible actions; explicit human request increases needs_human and escalation; resolved selects close_no_action; missing info selects request_information. Compare original/shared/full edits to isolate pathway.\n")
```
