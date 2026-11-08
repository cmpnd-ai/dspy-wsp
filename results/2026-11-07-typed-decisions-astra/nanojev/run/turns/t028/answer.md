I’ll measure the controlled cue responses and inspect remaining development errors. This will show both what the readout repair genuinely fixes and where it still fails.
```python
# Hypothesis: cue sensitivity should improve specifically for semantic evidence, while numeric reconciliation may remain weak.
def plain_read(k):
    f=WORK/f"fwd_{k}"
    return json.loads((f/"meta.json").read_text()),{p.stem:np.load(p) for p in f.glob("*.npy")}
cue_meta,cue_orig=plain_read(10)
_,cue_edit=plain_read(11)
print("Backbone/head-input max change:",float(np.nanmax(np.abs(cue_orig["head_inputs"]-cue_edit["head_inputs"]))))
cue_groups=collections.defaultdict(list)
for j,m in enumerate(cue_meta["rows"]):
    c=cue_info[m["request"]]
    cue_groups[(c["kind"],m["question"],str(c["value"]))].append(j)
cue_summary=[]
for (kind,q,val),js in cue_groups.items():
    m=cue_meta["rows"][js[0]];opts=m["options"];n=len(opts)
    po=cue_orig["probabilities"][js,:n].mean(axis=0);pe=cue_edit["probabilities"][js,:n].mean(axis=0)
    if kind=="topic":
        idx=opts.index(val)
        text=f"{kind}/{val}: target P {po[idx]:.3f}->{pe[idx]:.3f}; top original={opts[po.argmax()]}, edited={opts[pe.argmax()]}"
    elif m["type"]=="score":
        text=f"{kind}/{q}/{val}: expected level {po@np.arange(n):.3f}->{pe@np.arange(n):.3f}"
    elif m["type"]=="noul":
        text=f"{kind}/{q}/{val}: P(true) {po[1]:.3f}->{pe[1]:.3f}"
    else:
        text=f"{kind}/{q}/{val}: {opts}; original {np.round(po,3).tolist()}; edited {np.round(pe,3).tolist()}"
    cue_summary.append(text)
print("\n".join(cue_summary))
# Actual per-question development results and confident failures for the validated candidate.
print("Representative remaining development failures:")
for wf in sorted({ex["inputs"]["workflow"] for ex in examples}):
    rr=sorted((r for r in r_full.values() if r["inputs"]["workflow"]==wf),key=lambda r:r["score"])
    r=rr[0]
    print(r["id"],r["score"],r["feedback"])
    print(json.dumps(r["inputs"]["state"],ensure_ascii=False)[:1300])
# A diagnostic ablation comparison must be paired over whole records.
note("Cue test forward10/11 (same hidden features):\n"+"\n".join(cue_summary))
note("Full minus prior-only on development150: gain0.105333 SE0.021463. Thus roughly half the full development gain requires the evidence component, not template priors alone.")
```
