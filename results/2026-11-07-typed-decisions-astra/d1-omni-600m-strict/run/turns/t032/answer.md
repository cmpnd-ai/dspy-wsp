Gain 4 produced the expected dose-response: invoice urgency reached 62%, versus 38% originally, and later-due invoices still chose routine. I’ll keep that candidate for now. A larger agent-trace failure is also clear: all 20 discovery traces with constraint violations were labeled “harmful,” but the model called all 20 “partial.” I’ll use matched counterfactuals to isolate the violation, irreversible-action, and tool-error signals.
```python
# Hypothesis: explicit agent execution counters are represented but underread for outcome, risk, and urgency.
note("Pass 8 payment gain4: discovery B .574 vs .557, paired +.0170 SE .00871 (41 improved,24 worse). Invoice urgency .38->.62; near-due elevated-routine margin +1.588 versus -.600 at gain2 and -2.789 original; later margin -1.978 and all 24 still routine. Checkpoint payment_gain4. Gain not yet independently confirmed.")
agent_seed_ids=[eid for eid in read_ids_a if by_id[eid]["inputs"]["workflow"]=="agent_trace_observability"][:16]
agent_variants=[
    ("clean", {"constraint_violations":0,"irreversible_actions":0,"tool_errors":0}, "checkpointed"),
    ("dry", {"constraint_violations":0,"irreversible_actions":0,"tool_errors":0}, "dry_run"),
    ("error1", {"constraint_violations":0,"irreversible_actions":0,"tool_errors":1}, "checkpointed"),
    ("error4", {"constraint_violations":0,"irreversible_actions":0,"tool_errors":4}, "checkpointed"),
    ("irreversible", {"constraint_violations":0,"irreversible_actions":2,"tool_errors":0}, "checkpointed"),
    ("violation1", {"constraint_violations":1,"irreversible_actions":2,"tool_errors":0}, "checkpointed"),
    ("violation3", {"constraint_violations":3,"irreversible_actions":2,"tool_errors":0}, "checkpointed"),
    ("uncontrolled", {"constraint_violations":1,"irreversible_actions":2,"tool_errors":1}, "unsupervised")
]
agent_requests=[]; agent_tags=[]
for eid in agent_seed_ids:
    req0=json.loads(base_by_id[eid]["calls"][0]["prompt"])
    for tag,counts,autonomy in agent_variants:
        req=copy.deepcopy(req0)
        st=req["state"]["inputs"]["state"]
        st["trace_summary"].update(counts)
        st["agent"]["autonomy"]=autonomy
        agent_requests.append(req)
        agent_tags.append({"id":eid,"variant":tag})
(WORK/"agent_counterfactuals.json").write_text(json.dumps({"type":"read","model":"original","requests":agent_requests,"layers":[16]}))
(WORK/"agent_counterfactual_tags.json").write_text(json.dumps(agent_tags))
note("## Pass 9 planned: matched agent counterfactuals\nCurrent edits remain payment gain4, but READ ORIGINAL ignores them. Original reads 16 discovery-A agent records x 8 controlled counter/autonomy variants =128 requests/640 questions (agent_counterfactuals.json). All other fields fixed per seed. Discovery finding: all 20 constraint-violation traces gold harmful, original partial; risk high only 10/20. Goal: isolate state cue directions for a targeted counter readout, not encode label frequency.")
print("Requests:",len(agent_requests),"questions:",5*len(agent_requests),"current edit hashes:",edit_manifest())
```
