The normalized inputs support a larger development gain than reweighting the existing GELU features. I’ll examine which questions benefit and identify two low-contribution scorer neurons that could carry the missing readout direction with minimal disturbance to the original computation.
```python
# Hypothesis: the scorer MLP suppresses a useful residual direction; two low-contribution GELU neurons can transmit it without altering the backbone.
dx=xsolutions[0.01]
delta_z=solutions[(0.03,0.003)]
print("Direct-input per-question effects:")
evx=evaluate_logits(D["logits"]+D["x"]@dx,details=True)
for k,(b,a) in evx["by_question"].items():
    print(k[0][:8],k[1],round(b,3),"->",round(a,3), "delta",round(a-b,3))
contrib_rms=((Zc[fitmask]*W0.squeeze())**2).mean(0).sqrt()
neuron_order=contrib_rms.argsort()
print("Least option-ranking-contributing GELU units:")
for k in neuron_order[:10]:
    j=int(k)
    print(j,"weight",float(W0[0,j]),"contrast_rms",float(contrib_rms[j]),"activation_mean",float(D["z"][fitmask][:,j].mean()))
print("Strongest normalized-input correction coordinates:",[(int(j),round(float(dx[j]),4)) for j in dx.abs().argsort(descending=True)[:12]])
# Compare a small number of closed-form target geometries; all use the same fit-only covariance and dev selection.
alt_solutions={}
pcenter=centered(D["gold"])
bprob=XX.T@((pcenter*D["temp"][:,None]/D["n"].sqrt()[:,None])[fitmask].double())
blogit=XX.T@((centered(D["logits"])/D["n"].sqrt()[:,None])[fitmask].double())
bpvec=Vx.T@bprob; blvec=Vx.T@blogit
for rel in [0.003,0.01,0.03,0.1]:
    lam=float(ex.sum()/1024)*rel
    dprob=(Vx@(bpvec/(ex+lam))).float()
    dlogit=(Vx@(blvec/(ex+lam))).float()
    for target_scale in [3,5,8]:
        d=target_scale*dprob-dlogit
        alt_solutions[(rel,target_scale)]=d
        l=D["logits"]+D["x"]@d
        print("prob target",rel,target_scale,"norm",round(float(d.norm()),2),
              "fit",round(evaluate_logits(l,"fit")["acc"],4),
              "dev",{k:round(v,5) for k,v in evaluate_logits(l).items()})
note("Input residual ridge (.01) dev 55.00% -> 64.82%, +9.82pp (selection set SE 1.66pp), Brier .05735 -> .03554. Final-GELU-only ridge peaks 62.14%. Hypothesis refined: fixed MLP loses/attenuates semantic option contrasts that remain in its normalized input. Will transmit a residual direction through an exact GELU +/- pair in two lowest-contribution neurons and test cue counterfactuals plus dose response before retaining.")
```
