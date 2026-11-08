The first correction helps, but its large weight norm suggests the scorer has compressed useful distinctions into weak directions. I’ll inspect that compression and compare a direct readout of the scorer’s normalized input with reweighting its existing GELU units.
```python
# Hypothesis: weak state-sensitive directions survive before the scorer MLP but are attenuated by its nearly low-rank feature map.
target_soft=centered(D["gold"].clamp_min(0.03).log()*D["temp"][:,None])
residual_soft=target_soft-centered(D["logits"])
ysoft=(residual_soft/D["n"].sqrt()[:,None])[fitmask].double()
vbsoft=V.T@(X.T@ysoft)
for rel in [0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003]:
    lam=float(eigs.sum()/1024)*rel
    delta=(V@(vbsoft/(eigs+lam))).float()
    delta-=torch.dot(delta,unit0)*unit0
    solutions[(0.03,rel)]=delta
    l=D["logits"]+D["z"]@delta
    print("GELU ridge",rel,"norm",round(float(delta.norm()),2),
          "fit",round(evaluate_logits(l,"fit")["acc"],4),
          "dev",{k:round(v,5) for k,v in evaluate_logits(l).items()})
sv=torch.linalg.svdvals(scorer_params["1.weight"])
print("scorer.1 singular values top:",sv[:10].tolist(),"median",float(sv.median()),"min",float(sv.min()))
Xc=centered(D["x"])
XX=(Xc/D["n"].sqrt()[:,None,None])[fitmask].double()
CX=XX.T@XX
ex,Vx=torch.linalg.eigh(CX)
print("Normalized input effective dimension:",float(ex.sum()**2/ex.square().sum()),"trace/d",float(ex.sum()/1024))
vbx=Vx.T@(XX.T@ysoft)
xsolutions={}
for rel in [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3]:
    lam=float(ex.sum()/1024)*rel
    dx=(Vx@(vbx/(ex+lam))).float()
    xsolutions[rel]=dx
    l=D["logits"]+D["x"]@dx
    print("INPUT ridge",rel,"norm",round(float(dx.norm()),3),
          "fit",round(evaluate_logits(l,"fit")["acc"],4),
          "dev",{k:round(v,5) for k,v in evaluate_logits(l).items()})
best_z_key=max(solutions,key=lambda k:evaluate_logits(D["logits"]+D["z"]@solutions[k])["acc"])
best_x_key=max(xsolutions,key=lambda k:evaluate_logits(D["logits"]+D["x"]@xsolutions[k])["acc"])
print("Best readout",best_z_key,"best input",best_x_key)
note("Initial ridge (log gold floor .03, relative penalty .01) improves dev 55.00% -> 61.34%, paired SE 1.45pp, but delta norm 9.05 vs original readout norm .353. Exploring scorer spectral attenuation and direct normalized-input residual before choosing a causal intervention. No edits saved.")
```
No edits are saved in this comparison.
