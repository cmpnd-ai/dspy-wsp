Question schema explains about 78% of query variation, yet the original head barely uses that variation to choose options. I’ll derive a joint low-rank key correction that uses four existing query directions to condition the evidence readout, then compare it with the simpler rank-one repair on development data.

```python
# Hypothesis: question-context query directions can condition the key readout and reduce interference between schemas.
rctx,popt=4,256
Uctx=qovec[:,-rctx:].float()
Uctx=Uctx-u[:,None]*(u@Uctx)[None,:]
Uctx,_=torch.linalg.qr(Uctx,mode="reduced")
gate_std=(fit["Q"]@Uctx/math.sqrt(768)).square().mean(0).sqrt()
Uctx=Uctx*(fit["g"].square().mean().sqrt()/gate_std)[None,:]
Popt=opvec[:,-popt:].float()
def joint_features(data):
    oc=center_options(data["o"],data["mask"])
    gates=data["Q"]@Uctx/math.sqrt(768)
    ctx=(gates[:,None,:,None]*(oc@Popt)[:,:,None,:]).flatten(2)/data["temp"][:,None,None]
    return torch.cat([x_for(data),ctx],dim=2)
Jfit=joint_features(fit)
Jdev=joint_features(dev)
JX=Jfit[fit["mask"]]*weights[:,None]
jy3=center_options(torch.log(fit["gold"].clamp_min(.1)),fit["mask"])-center_options(fit["L"]/fit["temp"][:,None],fit["mask"])
jy=jy3[fit["mask"]]*weights
t0=time.time()
jgram=JX.T.double()@JX.double()
jrhs=JX.T.double()@jy.double()
jeig,jvec=torch.linalg.eigh(jgram)
jproj=jvec.T@jrhs
print("Joint closed-form feature count",JX.shape[1],"eigensolve seconds",round(time.time()-t0,2))
joint_candidates={}
joint_results=[]
for lam in [.0003,.001,.003,.01]:
    coeff=(jvec@(jproj/(jeig+lam))).float()
    joint_candidates[lam]=coeff
    for strength in [.5,.75,1.]:
        lf=fit["L"]+strength*(Jfit@coeff)*fit["temp"][:,None]
        ld=dev["L"]+strength*(Jdev@coeff)*dev["temp"][:,None]
        sf,sd=accuracy_stats(fit,lf),accuracy_stats(dev,ld)
        inc=paired_between(dev,dev["L"]+Cdev_ref,ld)
        joint_results.append((lam,strength,sf,sd,inc))
        print(f"joint ridge={lam:g} s={strength:.2f}: fit={sf[0]:.4f} dev={sd[0]:.4f} "
              f"increment best2={inc[0]:+.4f} SE={inc[1]:.4f} Brier={sd[3]:.4f}")
best_joint=max(joint_results,key=lambda r:(r[3][0],-r[3][3]))
print("Best joint candidate",best_joint)
lognote("Joint key-readout experiment: one closed-form solve from ORIGINAL-head residual targets, "
        "not iterative residual fitting. Features = full 2048-dimensional mean-query key path plus "
        "4 orthogonal query context modes x 256 option PCs; fit-only subspaces, log-gold floor .1. "
        "All candidates derived independently from same sufficient statistics. Results:\n"+
        "\n".join(f"- lambda={l}, strength={s}, fit={sf[0]:.4f}, dev={sd[0]:.4f}, "
                  f"inc_best2={inc[0]:+.4f} SE={inc[1]:.4f}, Brier={sd[3]:.4f}"
                  for l,s,sf,sd,inc in joint_results))
```
