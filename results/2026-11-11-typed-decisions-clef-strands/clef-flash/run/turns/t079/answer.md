The metadata check matches the local calculation. I’ll use two remaining passes to test cue responses on different practice seeds that were excluded from fitting this selected candidate, then finish with a full practice audit and final validation. No further weight fitting is planned.

```python
# Hypothesis: the selected repair's cue-sensitive effects generalize to seeds excluded from its fitting set.
mm21,ff21,mk21,ll21=read_probe_arrays(21)
assert mm21==nuis_mm
print("Metadata probe feature error:",float((ff21-nuis_ff).abs().max()))
print("Metadata probe logit reconstruction error:",float((ll21-nuis_l2)[nuis_mk].abs().max()))
held_probe_requests=[]
held_probe_meta=[]
held_seed_ids={wn:[ii for ii in splits["test"] if byid[ii]["inputs"]["workflow"]==wn][10:20] for wn in repair_workflows}
def add_held_probe(seed,family,level,rq):
    held_probe_requests.append(rq)
    held_probe_meta.append({"seed":seed,"family":family,"level":level})
for ii in held_seed_ids["agent_trace_observability"]:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for family,key,levels in [("tool_errors","tool_errors",[0,1,2,4]),
                              ("constraint_violations","constraint_violations",[0,1,2])]:
        for level in levels:
            rq=copy.deepcopy(rq0)
            ss=rq["state"]["inputs"]["state"]["trace_summary"]
            ss.update(tool_errors=0,constraint_violations=0,irreversible_actions=0)
            ss[key]=level
            add_held_probe(ii,family,level,rq)
for ii in held_seed_ids["customer_service"]:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for level,text in customer_texts.items():
        rq=copy.deepcopy(rq0)
        rq["state"]["inputs"]["state"]["thread"]=[{"role":"customer","text":text}]
        add_held_probe(ii,"customer_resolution_authority",level,rq)
for ii in held_seed_ids["invoice_processing"]:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for level in [False,True]:
        rq=copy.deepcopy(rq0)
        ss=rq["state"]["inputs"]["state"]
        invoice_id=ss["invoice"]["id"]
        prior_ids=[x for x in ss["vendor_history"]["prior_invoice_ids"] if x!=invoice_id]
        while len(prior_ids)<3: prior_ids.append(f"INV-2025-{7211+len(prior_ids)*19}")
        prior_ids=prior_ids[:3]
        if level: prior_ids[0]=invoice_id
        ss["vendor_history"]["prior_invoice_ids"]=prior_ids
        add_held_probe(ii,"duplicate_invoice_id",level,rq)
for ii in held_seed_ids["security_incidents"]:
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for level,allowlisted,window in [("unapproved",False,False),("allowlisted",True,False),("allowlisted_change_window",True,True)]:
        rq=copy.deepcopy(rq0)
        rq["state"]["inputs"]["state"]["context"].update(source_on_allowlist=allowlisted,change_window_active=window)
        add_held_probe(ii,"benign_security_context",level,rq)
assert len(held_probe_requests)==170
for model in ["original","edited"]:
    json.dump({"type":"read","model":model,"requests":held_probe_requests},open(WORK/f"held_cue_probes_{model}.json","w"))
json.dump(held_probe_meta,open(WORK/"held_cue_probe_meta.json","w"))
print("Independent-seed probes:",len(held_probe_requests),"requests,",len(held_probe_requests)*5,"questions")
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 21 verification\nMetadata edited run matched scorer simulation; upstream features unchanged. Positive descriptive accuracy differences persisted under metadata perturbations, but edited probabilities were somewhat more sensitive. Not claiming invariance.\n")
    f.write("\n## Planned pass 22 — independent-seed cue probes\nExact edits best_2 unchanged. READ original on170 requests/850 questions from40 practice seeds excluded from best_2's fit and tune (10/workflow, test positions10:20). Agent tool_errors0/1/2/4, violations0/1/2; customer5 authority/resolution states; invoice duplicate-ID absent/present at fixed prior-list length; security unapproved/allowlisted/allowlisted+window. No gold supplied, no fitting, prespecified margins from earlier probes. Files held_cue_probes_original.json and held_cue_probe_meta.json.\n")
```
