The tied readout has nine independent option contrasts, and the observed temperature is consistently 1.3374. I’ll now compare closed-form ridge solutions and intervention strengths using only the final write, its two RMS normalizations, and the tied readout. This predicts whether the repair generalizes before spending a full program pass.
```python
# Hypothesis: a regularized, rank-nine redirection improves held-out option selection while preserving unrelated residual directions.
Bt=B.T.copy()
gn64=gn_np.astype(np.float64); gp64=gp_np.astype(np.float64)
def recompute_head(d,A,alpha=1.0):
    u=d["u"].astype(np.float64)
    coeff=u@A.T
    unew=u+alpha*(coeff@Bt)
    # Only final residual-write normalization, residual addition, final norm, and tied readout.
    unew*=np.sqrt((u*u).sum(1)/(unew*unew).sum(1))[:,None]
    t=d["h"].astype(np.float64)/gn64
    tnew=t+(unew-u)*gp64
    hnew=tnew/np.sqrt(np.mean(tnew*tnew,axis=1))[:,None]*gn64
    logits=np.einsum("nkd,nd->nk",d["e"],hnew)
    logits[~d["mask"]]=-1e8
    scaled=logits/d["temp"][:,None]
    p=np.exp(scaled-scaled.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    return logits,p
def metrics_array(d,p):
    ok=(p.argmax(1)==d["gold"].argmax(1)).astype(float)
    brier=((p-d["gold"])**2).sum(1)
    baseok=(d["p"].argmax(1)==d["gold"].argmax(1)).astype(float)
    per=collections.defaultdict(list)
    for eid,val in zip(d["row_ids"],ok-baseok):per[eid].append(val)
    paired=np.array([np.mean(v) for v in per.values()])
    return dict(acc=float(ok.mean()), brier=float(brier.mean()),
                gain=float(paired.mean()), se=float(paired.std(ddof=1)/np.sqrt(len(paired))))
basefit=metrics_array(fit,fit["p"]); basedev=metrics_array(dev,dev["p"])
print("Baseline fit/dev",basefit,basedev)
VY=evecs.T@Yt
UtV=U.T@evecs
ridge_lambdas=[0.1,1.,10.,30.,100.,300.,1000.,3000.,10000.]
alphas=[0.5,0.75,1.,1.25,1.5]
A_by_lambda={}
grid=[]
for lam in ridge_lambdas:
    A=(UtV@(VY/(evals.clamp_min(0)+lam)[:,None])).T.numpy()
    A_by_lambda[lam]=A
    for alpha in alphas:
        _,pd=recompute_head(dev,A,alpha)
        md=metrics_array(dev,pd)
        grid.append((md["acc"],-md["brier"],lam,alpha,md))
    bestlam=max([g for g in grid if g[2]==lam])
    _,pf=recompute_head(fit,A,bestlam[3])
    print(f"ridge {lam:7g} dev best alpha={bestlam[3]:.2f} acc={bestlam[0]:.4f} brier={-bestlam[1]:.4f} fit_acc={metrics_array(fit,pf)['acc']:.4f}")
print("Top development settings:")
for row in sorted(grid,reverse=True)[:10]:print(row[2:])
best_grid=max(grid)
best_lam,best_alpha=best_grid[2:4]
A_best=A_by_lambda[best_lam]
l_dev,p_dev=recompute_head(dev,A_best,best_alpha)
print("Selected preliminary",best_lam,best_alpha,metrics_array(dev,p_dev))
for key in sorted(set(dev["keys"])):
    ix=np.array([k==key for k in dev["keys"]])
    old=(dev["p"][ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    new=(p_dev[ix].argmax(1)==dev["gold"][ix].argmax(1)).mean()
    print(f"{key[0]:27s}/{key[1]:23s} {old:.3f} -> {new:.3f}")
np.savez(WORK/"rank9_closed_form.npz",B=B,A=A_best,lambda_=best_lam,alpha=best_alpha)
note("Head-only development grid (not actual model passes):\n"+"\n".join(f"ridge={lam}, best="+str(max([g for g in grid if g[2]==lam])[3:]) for lam in ridge_lambdas)+f"\nPreliminary setting ridge={best_lam}, alpha={best_alpha}. Saved factors to rank9_closed_form.npz. Current edit: none.")
```
