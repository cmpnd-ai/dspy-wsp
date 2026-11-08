I’ll break down the holdout results by question and workflow, then compare original and edited readouts on the same records. This will show where the gain occurs and whether the edit changes option rankings rather than merely confidence.

```python
# Hypothesis: gains are concentrated in particular decisions and involve changed rankings.
from collections import defaultdict
import re
byq=defaultdict(list); byw=defaultdict(list)
for a,b in zip(r17o,r17e):
    assert a["id"]==b["id"]
    wf=a["inputs"]["workflow"]
    byw[wf].append(b["score"]-a["score"])
    for fa,fb in zip(a["feedback"].split("; "),b["feedback"].split("; ")):
        q=fa.split(":")[0]
        byq[(wf,q)].append(int(": right " in fb)-int(": right " in fa))
def stat(x):
    return (len(x),round(float(np.mean(x)),4),round(float(np.std(x,ddof=1)/np.sqrt(len(x))),4))
print("Workflow n,gain,SE:", {k:stat(v) for k,v in byw.items()})
print("Question n,gain,SE:",{"/".join(k):stat(v) for k,v in byq.items()})
print("Example changes:",r17o[0]["feedback"],"\n",r17e[0]["feedback"])
print("Metadata array specs:",{k:v for k,v in json.loads((work/"fwd_2"/"meta.json").read_text()).items() if k!="rows"})
audit_ids=[r["id"] for r in r17o]
for j in range(3):
    for model in ["original","edited"]:
        req={"type":"read","model":model,"from":{"pass":17,"model":"original","ids":audit_ids[j*150:(j+1)*150]}}
        json.dump(req,(work/f"audit_{j}_{model}.json").open("w"))
with (work/"notes.md").open("a") as f:
    f.write("Pass19 diagnostic READ original, first150 pass17 holdout records (750 questions). Validated C-strength2 stays installed; original READ bypasses edits. No fitting.\n")
```
