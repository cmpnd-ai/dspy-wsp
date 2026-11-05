The probes support stronger readout sensitivity for ownership, severity, urgency, and especially workflow impact. They also expose limits: intent barely responds, and single-sensor probes do not cross the decision boundary. I’ll now check a metric-aligned correction: each sensor-only record receives the same total weight as each three-question record, rather than one-third as much.

```python
# Hypothesis: weighting each record equally improves sensor evidence recovery without sacrificing other workflows.
dw=np.array([np.sqrt(3) if byid[rid]["inputs"]["workflow"]=="partial_observation_calibration" else 1. for rid in design_ids])
XW=XX*dw[:,None]; YW=YY*dw
weightedcv={}
for lam in [.03,.1,.3,1]:
    weightedcv[lam]={}
for fold in range(3):
    mask=np.array([foldmap[rid]!=fold for rid in design_ids])
    e,u=np.linalg.eigh(XW[mask].T@XW[mask]); ry=u.T@(XW[mask].T@YW[mask])
    held={rid for rid in foldmap if foldmap[rid]==fold}
    for lam in weightedcv:
        dd=u@(ry/(np.maximum(e,0)+lam))
        weightedcv[lam].update(eval_all(dd,held)[0])
for lam,v in weightedcv.items():
    diff=np.array([v[rid]-cv[.1][rid] for rid in v])
    print("weighted",lam,"CV",np.mean(list(v.values())),"gain vs unweighted .1",diff.mean(),"SE",diff.std(ddof=1)/np.sqrt(len(diff)))
# Leave confirmed best active for a complete practice benchmark.
with open(notes,"a") as f:
    f.write("\nPass 13: confirmed best scalar=original+delta900 ridge.1; decide edited on all 900 practice for complete baseline comparison. Counterfactual failures remain in intent and one-sensor OOD probes; no claim of complete reasoning repair.\n")
json.dump({"type":"decide","model":"edited","problems":[r["id"] for r in records]},open(work/"all_edited.json","w"))
```
