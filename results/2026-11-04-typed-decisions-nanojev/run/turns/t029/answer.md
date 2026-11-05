Across all 300 test cases, fixed offsets explain part of the gain (36.2% → 51.2%), but state-sensitive corrections add another 6.27 points. I’ll now construct controlled pairs that differ only in the decisive cue, keeping the validated weights unchanged.

```python
# Hypothesis: the repaired scalar direction responds causally to category text and explicit trace violations.
import copy
recbyid={r["id"]:r for r in records}
synthetic=[]; synlabels=[]
base=json.loads(recbyid["train_222"]["calls"][0]["prompt"])
for violations in [0,1,3]:
    rr=copy.deepcopy(base)
    rr["state"]["inputs"]["state"]["trace_summary"].update(constraint_violations=violations,irreversible_actions=0,tool_errors=0)
    synthetic.append(rr);synlabels.append(f"agent violations={violations}")
base=json.loads(recbyid["train_566"]["calls"][0]["prompt"])
for category,text in [
    ("billing","I was charged twice for the same subscription. Please correct the duplicate charge."),
    ("technical","The app crashes whenever I try to sign in. Please help troubleshoot the software error."),
    ("delivery","My package has not arrived and the tracking shows it is stuck in transit."),
    ("account","I need to update the email address and contact details on my account.")]:
    rr=copy.deepcopy(base)
    rr["state"]["inputs"]["state"]["thread"]=[{"role":"customer","text":text}]
    synthetic.append(rr);synlabels.append("customer "+category)
json.dump({"type":"read","model":"original","requests":synthetic},open(work/"synthetic_original.json","w"))
json.dump({"type":"read","model":"edited","requests":synthetic},open(work/"synthetic_edited.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nFixed offsets .512 vs full .574667 on 300 test cases: state-dependent correction adds .062667 accuracy.\n"
            "Passes 11–12: original and validated edited reads of 7 controlled cue variants (3 trace violation counts, 4 customer categories). No fitting or weight changes.\n")
```
