I’ll quantify the existing counterfactual results and workflow-level gains. The counterfactuals hold the surrounding record fixed while changing violation counts or invoice due dates, making them more informative than comparing unrelated right and wrong cases.

```python
# Hypothesis: the repaired scorer responds more strongly to violations and invoice timing while retaining unchanged hidden representations.
p0=np.load(work/"fwd_11"/"probabilities.npy")
p1=np.load(work/"fwd_12"/"probabilities.npy")
h0=np.load(work/"fwd_11"/"head_inputs.npy")
h1=np.load(work/"fwd_12"/"head_inputs.npy")
print("Head input max difference",np.nanmax(np.abs(h0-h1)))
for qi in ["needs_review","outcome","urgency"]:
    print("\n",qi)
    for r in cm["rows"]:
        if r["question"]!=qi: continue
        i=cm["rows"].index(r); n=len(r["options"])
        print(r["request"],dict(zip(r["options"],np.round(p0[i,:n],2))),"->",dict(zip(r["options"],np.round(p1[i,:n],2))))
from collections import defaultdict
import re
agg=defaultdict(list)
for passid in [13,14]:
    aa=[json.loads(s) for s in open(work/f"gen_{passid}_original.jsonl")]
    bb=[json.loads(s) for s in open(work/f"gen_{passid}_edited.jsonl")]
    for a,b in zip(aa,bb):
        aaok=re.findall(r"(\w+): (right|WRONG)",a["feedback"])
        bbok=dict(re.findall(r"(\w+): (right|WRONG)",b["feedback"]))
        for q,ok in aaok:
            agg[(a["inputs"]["workflow"],q)].append((ok=="right",bbok[q]=="right"))
for k,v in sorted(agg.items()):
    a=np.array(v,dtype=float); d=a[:,1]-a[:,0]
    print(k, "n",len(a),"base/new",np.round(a.mean(0),3),"gain/se",round(d.mean(),3),round(d.std(ddof=1)/np.sqrt(len(d)),3))
print("Cue file",json.load(open(work/"cues_edited.json")).keys())
```
