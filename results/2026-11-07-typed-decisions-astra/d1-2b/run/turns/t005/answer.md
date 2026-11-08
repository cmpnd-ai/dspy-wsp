The largest error is customer-service action: 100 of 225 records that should escalate instead request more information, often after the customer has already supplied it. There are also question-specific severity and review biases, so a global temperature change would not fix them. I’ll read the final decision inputs on a balanced fitting subset to see whether the last MLP contains usable evidence that its output is not expressing.
```python
# Hypothesis: question- and state-dependent evidence reaches the last MLP but is poorly aligned with the tied option-token readout.
read_groups = {}
for group, start in [("fit_a",0),("fit_b",45)]:
    read_groups[group] = []
    for wf in sorted({e["inputs"]["workflow"] for e in examples}):
        wfids = [eid for eid in split["fit"] if by_id[eid]["inputs"]["workflow"]==wf]
        read_groups[group] += wfids[start:start+45]
read_groups["dev"] = split["dev"]
for group,ids in read_groups.items():
    (WORK/f"read_{group}.json").write_text(json.dumps({"type":"read", "model":"original",
        "from":{"pass":1,"ids":ids}, "layers":[23,24]}))
print("Read groups:", {k:len(v) for k,v in read_groups.items()})
print("Customer-service action criteria:", examples[225]["labels"]["questions"] if examples[225]["inputs"]["workflow"]=="customer_service" else
      next(e["labels"]["questions"] for e in examples if e["inputs"]["workflow"]=="customer_service"))
for key in [("security_incidents","credential_compromise"), ("invoice_processing","discrepancy_severity")]:
    print("\nHigh-confidence errors",key)
    for conf,eid in sorted(bad_records[key], reverse=True)[:1]:
        r=base_by_id[eid]
        print(eid,json.dumps(r["inputs"]["state"])[:2200])
        print("pred",call_probs(r)[key[1]],"gold",by_id[eid]["labels"]["gold"][key[1]]["probabilities"])
note("Mechanism hypothesis: final-block state/question features are available but their option-token contrasts are misaligned. Largest error: customer-service action, 100/225 gold escalation -> request_information. Errors also include agent review over-triggering, security credential over-triggering, and question-specific score shifts. Will test a low-rank last-MLP output correction, not temperature or protected tensors.\n\nPlanned pass 2: READ original, 180 balanced fit_a records (900 questions), IDs in read_fit_a.json, layers 23 and 24. Current edit: none. Will inspect actual head-input arrays before deriving any intervention.")
```
