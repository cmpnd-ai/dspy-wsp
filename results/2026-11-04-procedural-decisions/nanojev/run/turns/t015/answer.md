Full strength raises practice accuracy to 66.89%, a gain of 19.70 points (paired SE 2.30). I’ll preserve this candidate, inspect which questions improved, and submit it to the hidden validation check before considering further changes.

```python
# Hypothesis: full-strength gains arise from evidence-sensitive decisions, not merely switching every boolean to false.
def summarize_file(path):
    rec=[json.loads(s) for s in open(path)]
    st=collections.defaultdict(list); preds=collections.defaultdict(collections.Counter)
    for rr in rec:
        a=json.loads(rr["calls"][0]["completion"])
        for q,g in byid[rr["id"]]["labels"]["gold"].items():
            p=a[q].get("probabilities",{"false":1-a[q].get("noul",0),"true":a[q].get("noul",0)})
            z=max(p,key=p.get); t=max(g["probabilities"],key=g["probabilities"].get)
            st[q].append(z==t); preds[q][z]+=1
    return {q:round(np.mean(v),4) for q,v in st.items()},dict(preds)
for k in [3,4]:
    print("pass",k,summarize_file(work/f"gen_{k}_edited.jsonl"))
import shutil
shutil.copytree(work/"edited",work/"best_full_ridge1",dirs_exist_ok=True)
with open(notes,"a") as f:
    f.write("\nPass 4 full strength disjoint 450: .47185 -> .66889, gain .19704 ± .02296 paired SE; 207 improved,76 worsened. Saved best_full_ridge1."
            "\nPass 5 hidden validation: same full-strength scalar edit; no other tensors changed.\n")
```
