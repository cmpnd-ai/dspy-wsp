I’ll evaluate the fixed readout corrections on the untouched features and compare several intervention strengths. This tests whether the direction changes the predicted behavior, rather than just fitting the original errors.

```python
# Hypothesis: correcting the final scorer direction generalizes, with a measurable dose-response in suppressed options.
def readset(k, ids):
    root=work/f"fwd_{k}"
    rr=json.load(open(root/"meta.json"))["rows"]
    hh=np.load(root/"head_inputs.npy")
    aa=np.nan_to_num(features(hh).numpy())
    ll=np.nan_to_num(np.load(root/"option_logits.npy"))
    mm=np.load(root/"option_mask.npy").astype(bool)
    order=[r["id"] for r in base if r["id"] in set(ids)]
    yy=np.zeros_like(ll)
    for i,r in enumerate(rr):
        g=byid[order[r["request"]]]["labels"]["gold"][r["question"]]["probabilities"]
        yy[i,:len(r["options"])]=[g[o] for o in r["options"]]
    return rr,aa,ll,mm,yy,order
rt,At,Lt,Mt,yt,test_order=readset(3,testids)
def measure(logits, rr, mm, yy):
    ok=(np.where(mm,logits,-1e10).argmax(1)==yy.argmax(1))
    qs=collections.defaultdict(list); items=collections.defaultdict(list)
    for i,r in enumerate(rr): qs[r["question"]].append(ok[i]); items[r["request"]].append(ok[i])
    scores=np.array([np.mean(v) for v in items.values()])
    return scores,{q:round(float(np.mean(v)),3) for q,v in qs.items()}
bt,btq=measure(Lt,rt,Mt,yt)
print("test baseline",bt.mean(),btq)
for reg,d in deltas.items():
    for strength in [.5,1.,1.5]:
        st,sq=measure(Lt+strength*(At@d),rt,Mt,yt)
        diff=st-bt
        print(reg,strength,round(st.mean(),4),"gain",round(diff.mean(),4),"SE",round(diff.std(ddof=1)/np.sqrt(len(diff)),4),sq)
```
