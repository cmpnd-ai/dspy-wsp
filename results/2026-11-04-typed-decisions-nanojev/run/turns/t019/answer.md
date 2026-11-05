The stronger intervention also matches exactly: accuracy rises to 53.1%, a gain of 16.8 ± 1.90 percentage points. I’ll inspect its remaining errors and compare a few closed-form target constructions within the fitting split, then freeze a repair before testing the untouched 300 practice cases.

```python
# Hypothesis: remaining errors reflect weak separation of gold-preferred candidates, rather than inadequate global confidence.
r6=[json.loads(s) for s in (work/"gen_6_edited.jsonl").read_text().splitlines()]
print("Internal repaired performance:")
s6=collections.defaultdict(list)
for r in r6:
    comp=json.loads(r["calls"][0]["completion"])
    for q,g in byid[r["id"]]["labels"]["gold"].items():
        o=comp[q]; p=o.get("probabilities",{"false":1-o.get("noul",0),"true":o.get("noul",0)})
        s6[(r["inputs"]["workflow"],q)].append(max(p,key=p.get)==max(g["probabilities"],key=g["probabilities"].get))
for k,v in s6.items(): print(k,round(np.mean(v),3))
# Alternative targets are solved independently in closed form, never via iterative loss optimization.
groups400=designs[0][2]+designs[1][2]
def target_residual(groups, mode):
    yy=[]
    for h,z,p,r in groups:
        n=len(p)
        if mode=="log": t=np.log(np.maximum(p,.015))
        elif mode=="linear": t=4*p
        elif mode=="winner": 
            t=np.zeros(n); t[p.argmax()]=2
        elif mode=="blend":
            t=np.log(np.maximum(p,.015)); t[p.argmax()]+=1
        yy.append(((t-t.mean())-(z-z.mean()))/np.sqrt(n))
    return np.concatenate(yy)
comparison=[]
for mode in ["log","linear","winner","blend"]:
    rr=X.T@target_residual(groups400,mode)
    for lam in [.01,.03,.1,.3,1]:
        dd=U@((U.T@rr)/(evals+lam))
        for a in [.5,.75,1]:
            sc,br=eval_dw(dd,designs[2][2],a)
            comparison.append((sc,br,mode,lam,a,dd))
for row in sorted(comparison,key=lambda t:-t[0])[:12]: print(row[:5])
with open(work/"notes.md","a") as f:
    f.write("\nPass 6 dose .75 confirmed: .363 -> .531, gain .168 ± .01899; 115 improved, 34 worsened. Scalar intervention reproduces prediction exactly at two doses.\n")
```
