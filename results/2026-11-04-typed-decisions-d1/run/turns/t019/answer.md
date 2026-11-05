Lower regularization recovers more fit-case rankings, but it could overfit. I’ll choose the regularization level with cross-validation entirely within the fit partition, preserving the reserved practice cases for the causal dose tests.

```python
# Hypothesis: low-variance contextual features generalize rather than merely fitting individual records.
fit_eids=sorted(fit_ids)
foldmap={eid:i%3 for i,eid in enumerate(fit_eids)}
cvresults={}
for ridge in [.01,.1,1,3,10]:
    vals=[]
    for fold in range(3):
        tr=fit & np.array([foldmap.get(r[0],-1)!=fold for r in rows])
        va=fit & ~tr
        a=torch.tensor(X[tr][M[tr]],dtype=torch.float64); yy=torch.tensor(Y[tr][M[tr]],dtype=torch.float64)
        dd=torch.linalg.solve(a.T@a+ridge*torch.eye(1024,dtype=torch.float64),a.T@yy).float().numpy()
        vals.append(report(dd,va)[0])
    cvresults[ridge]=np.mean(vals)
print("Fit-only three-fold accuracy:",cvresults)
chosen=max(cvresults,key=cvresults.get); delta=deltas[chosen]
print("Chosen ridge",chosen,"reserved test prediction",report(delta,~fit))
np.save(work/"readout_delta.npy",delta)
print("Largest changed neurons",sorted(enumerate(delta),key=lambda x:abs(x[1]),reverse=True)[:12])
for a in [.5,1,1.5]:
    print("dose",a,"fit",report(a*delta,fit),"test",report(a*delta,~fit))
save_edit("head.scorer.3.weight",torch.tensor(w0+.5*delta).reshape(1,-1))
json.dump({"type":"decide","model":"both","problems":sorted(test_ids)},open(work/"test_both.json","w"))
with notes.open("a") as f:
    f.write(f"\nFit-only 3-fold regularization selection: {cvresults}; selected ridge {chosen}. Current edit: head.scorer.3.weight = original + 0.5 * saved readout_delta.npy, no other edits. Next pass both on 450 reserved practice cases; dose-response plan .5, 1, 1.5.\n")
```
