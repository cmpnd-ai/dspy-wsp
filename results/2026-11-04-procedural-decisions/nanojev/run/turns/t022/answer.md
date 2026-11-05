I’ll combine the reads with verified ID alignment and compare closed-form corrections using three practice-only folds. This tests whether the weak intent/urgency results reflect limited fitting data or information that the scalar head cannot reliably recover.

```python
# Hypothesis: more fitting examples improve the evidence direction, while cross-fold estimates expose unrecoverable questions.
allrows=[]; hs=[]; zs=[]; gs=[]; idsrows=[]
for pn,ids in [(2,fit_ids),(7,read_groups[0]),(8,read_groups[1])]:
    mm=json.load(open(work/f"fwd_{pn}/meta.json"))
    aa=np.load(work/f"fwd_{pn}/head_inputs.npy").astype(float)
    zz=np.load(work/f"fwd_{pn}/option_logits.npy").astype(float)
    order=[r["id"] for r in records if r["id"] in set(ids)]
    for i,r in enumerate(mm["rows"]):
        rid=order[r["request"]]; g=byid[rid]["labels"]["gold"][r["question"]]["probabilities"]
        allrows.append(r); hs.append(aa[i]); zs.append(zz[i]); gs.append(np.array([g[o] for o in r["options"]]))
        idsrows.append(rid)
HH=np.stack(hs); ZZ=np.stack(zs)
XX=[]; YY=[]; design_ids=[]
for i,r in enumerate(allrows):
    k=len(gs[i])
    if r["type"]=="noul":
        xx=HH[i,1:2]; target=np.log(np.clip(gs[i][1],.1,.9)/np.clip(gs[i][0],.1,.9))
        yy=np.array([target-ZZ[i,1]])
    else:
        xx=(HH[i,:k]-HH[i,:k].mean(0))/np.sqrt(k)
        yy=(3*(gs[i]-gs[i].mean())-(ZZ[i,:k]-ZZ[i,:k].mean()))/np.sqrt(k)
    XX.append(xx); YY.append(yy); design_ids += [idsrows[i]]*len(yy)
XX=np.concatenate(XX); YY=np.concatenate(YY)
def eval_all(d, subset=None, strength=1):
    item=collections.defaultdict(list); qstats=collections.defaultdict(list)
    for i,r in enumerate(allrows):
        if subset is not None and idsrows[i] not in subset:continue
        k=len(gs[i]); change=HH[i,:k]@d*strength
        if r["type"]=="noul":change[0]=0
        ok=float(np.argmax(ZZ[i,:k]+change)==np.argmax(gs[i]))
        item[idsrows[i]].append(ok); qstats[r["question"]].append(ok)
    return {rid:np.mean(v) for rid,v in item.items()}, {q:round(np.mean(v),4) for q,v in qstats.items()}
foldmap={}
rng=random.Random(926)
for wf in sorted(set(x["inputs"]["workflow"] for x in data)):
    ids=[x["id"] for x in data if x["inputs"]["workflow"]==wf]; rng.shuffle(ids)
    foldmap.update({rid:j%3 for j,rid in enumerate(ids)})
cv={lam:{} for lam in [.1,.3,1,3]}
cvq={lam:collections.defaultdict(list) for lam in cv}
for fold in range(3):
    mask=np.array([foldmap[rid]!=fold for rid in design_ids])
    e,u=np.linalg.eigh(XX[mask].T@XX[mask]); ry=u.T@(XX[mask].T@YY[mask])
    held={rid for rid in foldmap if foldmap[rid]==fold}
    for lam in cv:
        dd=u@(ry/(np.maximum(e,0)+lam))
        vals, qs=eval_all(dd,held)
        cv[lam].update(vals)
        for q,v in qs.items():cvq[lam][q].append(v)
for lam,vals in cv.items():
    print("ridge",lam,"crossfold",round(np.mean(list(vals.values())),4),
          {q:round(np.mean(v),3) for q,v in cvq[lam].items()})
print("existing correction all",np.mean(list(eval_all(delta)[0].values())))
```
