Several errors are directional: agent runs are often called successful instead of partial, urgency is compressed toward middle levels, and security alerts over-predict credential compromise. I’ll inspect examples of those failures, then read head activations from 200 fitting records to see where the option margins go wrong.


This block records representative failures and prepares the first activation read. No weights will change.

```python
# Hypothesis: the final residual readout underweights state-specific evidence and favors stereotyped options.
print("Split sizes:", {k:len(v) for k,v in splits.items()})
print("Final head computations:\n", "\n".join(source.splitlines()[391:461]))
fitset = set(splits["fit"])
wanted = [
    ("agent_trace_observability", "outcome", "partial", "success"),
    ("agent_trace_observability", "urgency", "3", "2"),
    ("security_incidents", "credential_compromise", "false", "true"),
    ("invoice_processing", "duplicate", "true", "false"),
    ("customer_service", "action", "request_information", "escalate_to_human")
]
for wf,q,gt,pred in wanted:
    rr = [r for r in groups[wf,q] if r["id"] in fitset and r["target"]==gt and r["pred"]==pred]
    print("\nCONFUSION:", wf,q,gt,"->",pred)
    for r in rr[:2]:
        ex = byid[r["id"]]
        print(r["id"], json.dumps(ex["inputs"]["state"], ensure_ascii=False))
        print("Predicted:", r["probs"], "Gold:", ex["labels"]["gold"][q]["probabilities"])
read_chunks = [splits["fit"][:200], splits["fit"][200:400], splits["fit"][400:]+splits["tune"]]
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":read_chunks[0]}},
          (WORK/"read_fit_1.json").open("w"))
note("## Diagnosis\nBaseline largest issues: agent outcome 55.1% (73 partial→success), agent urgency 51.1% (54 low→routine, 28 critical→elevated), invoice urgency 51.6%, agent risk 59.1%, invoice disposition 60.0%, security credential compromise 67.1% (70 false→true). Hypothesis: state cues are available but final option-margin readout is misaligned, with workflow-specific stereotyped preferences. Will test final scorer before altering routing.\n## Planned pass 2\nExact edits: none. READ original on first 200 fit IDs (1000 questions), read_fit_1.json. Gold never supplied to model. No test examples.")
```
