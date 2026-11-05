I’ll summarize whether each fact change moves the correct answer’s margin in the expected direction, and whether both members of a pair are answered correctly. These are stronger tests than ordinary right/wrong feature correlations.

```python
# Hypothesis: scalar repair amplifies rule-relevant contrasts while some relational rules remain unrecovered.
probe_summary={}
for pn in [11,12]:
    mm=json.load(open(work/f"fwd_{pn}/meta.json"))
    ll=np.load(work/f"fwd_{pn}/option_logits.npy")
    ss=collections.defaultdict(list)
    for pair in probe_pairs:
        i,j=pair["rows"]; t0,t1=pair["targets"]
        opts=mm["rows"][i]["options"]; u,v=opts.index(t0),opts.index(t1)
        shift=(ll[j,v]-ll[j,u])-(ll[i,v]-ll[i,u])
        both=(opts[np.argmax(ll[i,:len(opts)])]==t0 and opts[np.argmax(ll[j,:len(opts)])]==t1)
        acc=.5*((opts[np.argmax(ll[i,:len(opts)])]==t0)+(opts[np.argmax(ll[j,:len(opts)])]==t1))
        ss[pair["question"]].append((float(shift),float(both),acc))
    probe_summary[pn]={q:np.mean(v,axis=0).round(3).tolist() for q,v in ss.items()}
print("Columns: mean correct-direction margin shift, fraction both correct, pair accuracy")
for q in stats: print(q,"original",probe_summary[11][q],"edited",probe_summary[12][q])
# Check original and edited feature invariance and exact head prediction.
PH=np.load(work/"fwd_11/head_inputs.npy")
PL=np.load(work/"fwd_11/option_logits.npy")
EL=np.load(work/"fwd_12/option_logits.npy")
pred=PL+np.einsum("nkd,d->nk",PH,delta900)
for i,r in enumerate(json.load(open(work/"fwd_11/meta.json"))["rows"]):
    if r["type"]=="noul":pred[i,0]=0
print("Max scalar intervention prediction error",np.nanmax(np.abs(pred-EL)))
with open(notes,"a") as f:
    f.write("\nCounterfactual probe summaries (margin shift,both correct,pair accuracy): "+str(probe_summary)+
            f"\nExact head intervention prediction max error {np.nanmax(np.abs(pred-EL)):.6f}.\n")
```
