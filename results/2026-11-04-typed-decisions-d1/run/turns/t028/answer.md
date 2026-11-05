The largest gains are in trace risk (+27.9 points), trace outcome (+19.2), churn risk (+18.1), and invoice urgency (+16.1). Mean per-option Brier also falls from 0.0576 to 0.0405. To test whether the edit strengthens relevant state cues, I’ll construct matched records that differ only in violation count or invoice due date.

```python
# Hypothesis: the repaired readout responds more strongly to violations and overdue-payment cues.
import copy
cue_requests=[]; cue_tags=[]
for eid in ["train_1","train_101","train_143"]:
    req=json.loads(oldrec[eid]["calls"][0]["prompt"])
    for violations in [0,1,2]:
        rr=copy.deepcopy(req)
        rr["state"]["inputs"]["state"]["trace_summary"]["constraint_violations"]=violations
        cue_requests.append(rr); cue_tags.append((eid,"violations",violations))
for eid in ["train_601","train_693"]:
    req=json.loads(oldrec[eid]["calls"][0]["prompt"])
    for days in [30,2,-21]:
        rr=copy.deepcopy(req)
        rr["state"]["inputs"]["state"]["payment"]["days_until_due"]=days
        rr["state"]["inputs"]["state"]["payment"]["status"]="overdue" if days<0 else "scheduled"
        cue_requests.append(rr); cue_tags.append((eid,"days_due",days))
for model in ["original","edited"]:
    json.dump({"type":"read","model":model,"requests":cue_requests},open(work/f"cues_{model}.json","w"))
with notes.open("a") as f:
    f.write("\nCue counterfactual reads planned original+edited: 9 trace requests (3 bases × violations 0/1/2) and 6 invoice requests (2 bases × due 30/2/-21). Current edit full-dose ridge1 final readout only. Purpose: test cue sensitivity independent of right/wrong-case difficulty.\n")
```
