I’ll prepare controlled probes that vary tool errors, constraint violations, duplicate invoice IDs, and security context while holding other fields fixed. These unscored probes will test whether the validated edit responds to evidence rather than merely changing option preferences.


```python
# Hypothesis: the validated correction strengthens responses to decisive state cues, not just fixed option biases.
import copy
print("Agent fit gold labels by tool errors, excluding constraint violations:")
agentfit=[byid[i] for i in splits["fit"] if byid[i]["inputs"]["workflow"]=="agent_trace_observability"]
for count in sorted({e["inputs"]["state"]["trace_summary"]["tool_errors"] for e in agentfit}):
    es=[e for e in agentfit if e["inputs"]["state"]["trace_summary"]["tool_errors"]==count and e["inputs"]["state"]["trace_summary"]["constraint_violations"]==0]
    print(count,len(es),dict(collections.Counter(e["labels"]["gold"]["outcome"]["label"] for e in es)))
probe_requests,probe_meta=[],[]
def add_probe(exid,state,family,level):
    req=json.loads(basebyid[exid]["calls"][0]["prompt"])
    req["state"]["inputs"]["state"]=state
    probe_requests.append(req)
    probe_meta.append({"seed":exid,"family":family,"level":level})
for ex in agentfit[:12]:
    eid=str(ex["id"])
    for errors in [0,1,2,4]:
        s=copy.deepcopy(ex["inputs"]["state"])
        s["trace_summary"].update(tool_errors=errors,constraint_violations=0,irreversible_actions=0)
        add_probe(eid,s,"agent_tool_errors",errors)
    for violations in [0,1,2]:
        s=copy.deepcopy(ex["inputs"]["state"])
        s["trace_summary"].update(tool_errors=0,constraint_violations=violations,irreversible_actions=1)
        add_probe(eid,s,"agent_constraint_violations",violations)
invoicefit=[byid[i] for i in splits["fit"] if byid[i]["inputs"]["workflow"]=="invoice_processing"]
for ex in invoicefit[:12]:
    eid=str(ex["id"]); inv=ex["inputs"]["state"]["invoice"]["id"]
    for dup in [False,True]:
        s=copy.deepcopy(ex["inputs"]["state"])
        hist=[v for v in s["vendor_history"].get("prior_invoice_ids",[]) if v!=inv]
        hist.append(inv if dup else "INV-2026-999999")
        s["vendor_history"]["prior_invoice_ids"]=hist
        add_probe(eid,s,"invoice_duplicate",dup)
securityfit=[byid[i] for i in splits["fit"] if byid[i]["inputs"]["workflow"]=="security_incidents"]
for ex in securityfit[:12]:
    eid=str(ex["id"])
    for level in ["original","allowlisted","allowlisted_change_window"]:
        s=copy.deepcopy(ex["inputs"]["state"])
        if level!="original":
            s["context"]["source_on_allowlist"]=True
        if level=="allowlisted_change_window":
            s["context"]["change_window_active"]=True
        add_probe(eid,s,"security_benign_context",level)
json.dump(probe_meta,(WORK/"causal_probe_meta.json").open("w"))
json.dump({"type":"read","model":"original","requests":probe_requests},(WORK/"causal_probes_original.json").open("w"))
json.dump({"type":"read","model":"edited","requests":probe_requests},(WORK/"causal_probes_edited.json").open("w"))
print("Controlled requests:",len(probe_requests),"questions:",5*len(probe_requests))
note("Minimum-margin alternatives did not beat best_1 on tune, so none saved.\n## Planned pass 8\nExact current edits: best_1 unchanged. READ original on 144 controlled requests (720 questions): 12 agent seeds × (4 tool-error levels + 3 violation levels), 12 invoice seeds × duplicate absent/present, 12 security seeds × 3 context conditions. Seeds are fit records only; probes unscored, no supplied gold. Files causal_probes_original.json and causal_probe_meta.json.")
```
