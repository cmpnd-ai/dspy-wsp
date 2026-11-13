The query-side addition gives only a small development improvement, so I won’t keep it without stronger evidence. I’ll first check the regularization boundary of the simpler, validated rank-one repair and test whether its treatment of very small gold probabilities is driving residual errors.

```python
# Hypothesis: the rank-one repair may be under-regularized in some directions or overly influenced by near-zero target probabilities.
def paired_between(data,La,Lb):
    pa=La.masked_fill(~data["mask"],-1e9).argmax(1)
    pb=Lb.masked_fill(~data["mask"],-1e9).argmax(1)
    target=data["gold"].argmax(1)
    diff=((pb==target).float()-(pa==target).float()).reshape(-1,5).mean(1)
    return diff.mean().item(),diff.std(unbiased=True).item()/math.sqrt(len(diff))
q_lam,q_s=bestq[:2]
qinc=paired_between(dev,dev["L"]+Cdev,
                    dev["L"]+Cdev+q_s*(Fdev@q_candidates[q_lam])*dev["temp"][:,None])
print("Best query incremental development gain/SE:",qinc)
print("No query edit saved; validated key-only edit remains active.")
refine=[]
refine_vs={}
for floor in [.003,.01,.03,.1]:
    yy3=center_options(torch.log(fit["gold"].clamp_min(floor)),fit["mask"])-center_options(fit["L"]/fit["temp"][:,None],fit["mask"])
    yy=yy3[fit["mask"]]*weights
    rr=evecs.T@(X.T.double()@yy.double())
    for lam in [.0001,.0003,.001,.003,.01]:
        vv=(evecs@(rr/(evals+lam))).float()
        refine_vs[(floor,lam)]=vv
        # Only compare a predeclared, compact strength set using the same closed-form direction.
        for ss in [.75,1.0]:
            lf=fit["L"]+ss*correction(fit,vv)
            ld=dev["L"]+ss*correction(dev,vv)
            sf,sd=accuracy_stats(fit,lf),accuracy_stats(dev,ld)
            refine.append((floor,lam,ss,sf,sd))
print("Top 12 refinement candidates by development accuracy:")
for floor,lam,ss,sf,sd in sorted(refine,key=lambda r:(r[4][0],-r[4][3]),reverse=True)[:12]:
    print(f"floor={floor:g} ridge={lam:g} s={ss:.2f}: fit={sf[0]:.4f} dev={sd[0]:.4f} "
          f"gain={sd[1]:+.4f} SE={sd[2]:.4f} Brier={sd[3]:.4f}")
best_ref=max(refine,key=lambda r:(r[4][0],-r[4][3]))
floor_ref,lam_ref,s_ref=best_ref[:3]
v_ref=refine_vs[(floor_ref,lam_ref)]*s_ref
print("Best refinement:",best_ref,"deltaK norm",v_ref.norm().item())
print("Increment vs validated key on dev:",paired_between(dev,dev["L"]+Cdev,dev["L"]+correction(dev,v_ref)))
lognote("Query residual candidate best development +.0100 over validated key, paired SE "+str(qinc[1])+
        "; no query weights saved. Not yet sufficient evidence to add complexity.")
lognote("Rank-one refinement uses the same fit-only eigensystem, predeclared floors .003/.01/.03/.1, "
        "ridge .0001/.0003/.001/.003/.01, strengths .75/1. No test/validation used to choose. "
        "Best development candidate="+str(best_ref)+". Still no new edit saved.")
```
