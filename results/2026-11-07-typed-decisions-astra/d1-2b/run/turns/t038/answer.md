The head calculation predicts that question-level bias correction alone reaches 71.1%, while restoring the state-dependent component reaches 78.3%. I’ll test that ablation in the actual model. In parallel, I’ll compare a small set of closed-form targets that put slightly more emphasis on the gold winning option, since the scored metric is accuracy.
```python
# Hypothesis: state-dependent directions explain gains beyond question-level biases; modest winner emphasis may better match the accuracy metric.
delta_question=(torch.from_numpy(B).double()@(torch.from_numpy(A_question).double()@w2orig.double())).float()
install_w2(1.,delta_question)
def contrast_targets_mixed(d,beta=0.,floor=.015):
    n=len(d["h"]); targets=np.zeros((n,rank),np.float32)
    for i in range(n):
        m=d["mask"][i]; E=d["e"][i,m].astype(np.float64)
        u=d["u"][i].astype(np.float64); t=d["h"][i].astype(np.float64)/gn64
        dt=gp64[:,None]*(B.astype(np.float64)-u[:,None]*(u@B)[None,:]/(u@u))
        dh=gn64[:,None]*(dt-t[:,None]*(t@dt)[None,:]/2048)
        J=E@dh; J-=J.mean(axis=0)
        gold=d["gold"][i,m].astype(np.float64)
        winner=gold.argmax()
        target_probs=(1-beta)*gold
        target_probs[winner]+=beta
        target=d["temp"][i]*np.log(np.maximum(target_probs,floor))
        target-=target.mean()
        orig=d["l"][i,m].astype(np.float64);orig-=orig.mean()
        targets[i]=np.linalg.pinv(J,rcond=1e-6)@(target-orig)
    return targets
target_candidates={}
target_grid=[]
for beta in [0.,.2,.5]:
    y=Yt if beta==0 else torch.from_numpy(contrast_targets_mixed(fit,beta)).double()
    for penalty,ev,V,xtv,chol in [("coefficient",evals,evecs,UtV,None),("weight",evz,vz,ZtV,L)]:
        vy=V.T@y
        for lam in [1.,3.,10.]:
            aa=(xtv@(vy/(ev.clamp_min(0)+lam)[:,None])).T
            if chol is not None:
                aa=torch.linalg.solve_triangular(chol.T,aa.T,upper=True).T
            aa=aa.numpy()
            target_candidates[(beta,penalty,lam)]=aa
            _,p=recompute_head(dev,aa)
            mm=metrics_array(dev,p)
            target_grid.append((mm["acc"],-mm["brier"],beta,penalty,lam,mm))
print("Closed-form target/penalty grid, top settings:")
for row in sorted(target_grid,reverse=True)[:12]:print(row[2:])
note("Planned pass 16: READ edited dev (candidate_dev.json), diagnostic state ablation. Exact edit W2_original+B A_question W2_original; A_question=A_ridge1 P, P orthogonal projector onto 20 fit question-mean write vectors. Thus within-question state-dependent complement is removed. Predicted accuracy .711111 versus .783333 with full edit. All norm/embedding tensors original. Confirmed full edit preserved in best_1/.\n\nIndependent closed-form target variants (fit only) considered beta=0,.2,.5 winner emphasis, coefficient or actual-weight norm ridge=1,3,10; dev scores:\n"+"\n".join(str(r[2:]) for r in sorted(target_grid,reverse=True)[:6]))
```
