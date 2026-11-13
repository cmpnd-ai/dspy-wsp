The mismatch was indeed read-export ordering; all rows now match the recorded requests to within \(3\times10^{-8}\) in probability. I’ll redo the closed-form experiment with correctly aligned labels. The invalid candidates were never saved.

```python
# Hypothesis: correctly aligned option contrasts support a generalizing rank-one key readout repair.
Xfit3=x_for(fit)
Xdev3=x_for(dev)
weights=(1/fit["mask"].sum(1).float()).sqrt()[:,None].expand_as(fit["mask"])[fit["mask"]]
X=Xfit3[fit["mask"]]*weights[:,None]
target_log=center_options(torch.log(fit["gold"].clamp_min(.01)),fit["mask"])
y3=target_log-center_options(fit["L"]/fit["temp"][:,None],fit["mask"])
y=y3[fit["mask"]]*weights
gram=X.T.double()@X.double()
rhs=X.T.double()@y.double()
evals,evecs=torch.linalg.eigh(gram)
projrhs=evecs.T@rhs
ridge_vs={}
results_ridge=[]
for lam in [.001,.01,.1,1.,10.,100.]:
    v=(evecs@(projrhs/(evals+lam))).float()
    ridge_vs[lam]=v
    for strength in [.5,1.0,1.5]:
        LF=fit["L"]+strength*(Xfit3@v)*fit["temp"][:,None]
        LD=dev["L"]+strength*(Xdev3@v)*dev["temp"][:,None]
        sf,sd=accuracy_stats(fit,LF),accuracy_stats(dev,LD)
        results_ridge.append((lam,strength,sf,sd))
        print(f"ridge={lam:7g} strength={strength:.1f} | fit={sf[0]:.3f} dev={sd[0]:.3f} "
              f"gain={sd[1]:+.3f} SE={sd[2]:.3f} Brier={sd[3]:.3f} | deltaK norm={v.norm().item()*strength:.2f}")
best_grid=max(results_ridge,key=lambda r:(r[3][0],-r[3][3]))
print("Best dev grid:",best_grid)
lam0,strength0=best_grid[:2]
v0=ridge_vs[lam0]
deltaK0=u[:,None]@v0[None,:]
group_order=sorted(set(fit["group"]))
def group_report(data,L):
    basepred=data["L"].masked_fill(~data["mask"],-1e9).argmax(1)
    pred=L.masked_fill(~data["mask"],-1e9).argmax(1)
    target=data["gold"].argmax(1)
    for g in sorted(set(data["group"])):
        ix=torch.tensor([i for i,x in enumerate(data["group"]) if x==g])
        old=(basepred[ix]==target[ix]).float().mean().item()
        new=(pred[ix]==target[ix]).float().mean().item()
        print(f"{g:51s} {old:.3f} -> {new:.3f} ({new-old:+.3f})")
print("\nDEV by question at selected grid point:")
group_report(dev,dev["L"]+strength0*(Xdev3@v0)*dev["temp"][:,None])
lognote("Correctly aligned rank-one key grid:\n"+"\n".join(
    f"- lambda={lam}, strength={s}: fit={sf[0]:.4f}, dev={sd[0]:.4f}, gain={sd[1]:+.4f} SE={sd[2]:.4f}, Brier={sd[3]:.4f}"
    for lam,s,sf,sd in results_ridge))
print("No edits saved yet.")
```
