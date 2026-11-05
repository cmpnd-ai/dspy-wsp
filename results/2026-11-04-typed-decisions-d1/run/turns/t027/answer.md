Validation confirms the edit: accuracy rises from 57.53% to 64.53%, a paired gain of 7.00 ± 1.30 percentage points. I’ll preserve that candidate and inspect which questions improved, which regressed, and whether probability quality also improved.

```python
# Hypothesis: the gain reflects repaired question-specific rankings, not global sharpening.
if (work/"best_1").exists():shutil.rmtree(work/"best_1")
shutil.copytree(work/"edited",work/"best_1")
newrec=[json.loads(s) for s in (work/"gen_8_edited.jsonl").read_text().splitlines()]
oldrec={r["id"]:r for r in records}
def distribution(ans):
    return ans.get("probabilities",{"false":1-ans.get("noul",0),"true":ans.get("noul",0)})
changes=collections.defaultdict(list); briers=[[],[]]
examples_changed=[]
for r in newrec:
    ex=byid[r["id"]]; oa=json.loads(oldrec[r["id"]]["calls"][0]["completion"]); na=json.loads(r["calls"][0]["completion"])
    for q,g in ex["labels"]["gold"].items():
        op=distribution(oa[q]); np_=distribution(na[q]); gg=g["probabilities"]
        target=max(gg,key=gg.get); p0=max(op,key=op.get); p1=max(np_,key=np_.get)
        changes[(ex["inputs"]["workflow"],q)].append(int(p1==target)-int(p0==target))
        for idx,p in enumerate([op,np_]): briers[idx].append(sum((p[o]-gg[o])**2 for o in gg)/len(gg))
        if p1==target and p0!=target:examples_changed.append((r["id"],q,p0,p1))
for k,v in changes.items():print(k,"gain",round(np.mean(v),3))
print("Mean per-option Brier",*[np.mean(b) for b in briers])
parallel=(delta@w0)/(w0@w0)*w0
perp=delta-parallel
print("Delta norm",np.linalg.norm(delta),"parallel norm",np.linalg.norm(parallel),"perp norm",np.linalg.norm(perp))
print("Parallel-only",report(parallel,~fit),"perp-only",report(perp,~fit))
for eid,q,p0,p1 in examples_changed[::max(1,len(examples_changed)//5)][:5]:
    print(eid,q,p0,"→",p1,json.dumps(byid[eid]["inputs"]["state"])[:1400])
with notes.open("a") as f:
    f.write("\nPass10 VALIDATION confirms full-dose: .57533→.64533, gain .07000±.01302, 133 improved/63 worsened. Saved best_1 and candidate_full. Current best = ridge1 full-dose final scorer weight only.\n")
```
