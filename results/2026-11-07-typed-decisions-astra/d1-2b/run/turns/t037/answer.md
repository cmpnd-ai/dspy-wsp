Validation confirms the repair: **60.67% → 75.73%**, a **15.07 ± 1.45 percentage-point** paired gain. I’ll preserve that edit now. Next I’ll test a more conservative closed-form variant that penalizes the actual weight change, and separate question-level bias correction from state-dependent evidence to understand what is driving the gains.
```python
# Hypothesis: minimum-weight-change regularization preserves the causal repair while reducing unnecessary changes outside observed decision states.
best_dir=WORK/"best_1"
if best_dir.exists():shutil.rmtree(best_dir)
shutil.copytree(WORK/"edited",best_dir)
best_confirmed={"name":"rank9_ridge1","directory":str(best_dir),"test_gain":.15666666666666668,
                "test_se":.01334145527929241,"validation_gain":.15066666666666667,
                "validation_se":.014504013463187339,"validation_score":.7573333333333334,
                "A":A_selected.copy(),"B":B.copy(),"alpha":1.0}
note("Pass 15 VALIDATION CONFIRMED, n=300: original .606667 -> edited .757333; paired gain +.150667 SE .014504; 171 improved, 43 worsened. First hidden validation query. Confirmed best copied to best_1/. Exact edit: rank9 ridge1 alpha1 W2 only. Independent practice gain +.156667 SE .013341.\n\nNext analysis (no forward yet): compare closed-form penalties on actual ||delta W2|| rather than ||A||, using only fit coefficients and dev selection; also decompose state-dependent versus question-centroid components. Best_1 remains safe.")
# A penalty A R A^T with R=W2 W2^T equals the Frobenius penalty on the actual low-rank delta W2.
R=w2orig.double()@w2orig.double().T
R+=torch.eye(2048,dtype=torch.float64)*1e-8
L=torch.linalg.cholesky(R)
Z=torch.linalg.solve_triangular(L,U.T,upper=False).T
Gz=Z@Z.T
evz,vz=torch.linalg.eigh(Gz)
ZY=vz.T@Yt
ZtV=Z.T@vz
A_weight={}
weight_grid=[]
for lam in [0.1,0.3,1.,3.,10.,30.,100.,300.]:
    Az=(ZtV@(ZY/(evz.clamp_min(0)+lam)[:,None])).T
    Az_orig=torch.linalg.solve_triangular(L.T,Az.T,upper=True).T.numpy()
    A_weight[lam]=Az_orig
    for alpha in [.75,1.,1.15]:
        _,p=recompute_head(dev,Az_orig,alpha)
        mm=metrics_array(dev,p)
        weight_grid.append((mm["acc"],-mm["brier"],lam,alpha,mm))
    bb=max(g for g in weight_grid if g[2]==lam)
    normdelta=np.sqrt(np.trace(Az_orig@R.numpy()@Az_orig.T))/float(w2orig.double().norm())
    print("weight-norm ridge",lam,"relative update",round(normdelta,4),"best",bb[3:])
print("Best weight-norm options",sorted(weight_grid,reverse=True)[:5])
# Orthogonal projector on the 20 question centroids; complementary activity is within-question state evidence.
keys_sorted=sorted(set(fit["keys"]))
M=np.stack([fit["u"][np.array([k==key for k in fit["keys"]])].mean(axis=0) for key in keys_sorted]).astype(np.float64)
_,_,Vm=np.linalg.svd(M,full_matrices=False)
A_question=(A_selected@Vm.T)@Vm
A_state=A_selected-A_question
for state_strength in [0.,.5,1.,1.5]:
    _,p=recompute_head(dev,A_question+state_strength*A_state)
    print("State-component strength",state_strength,metrics_array(dev,p))
```
