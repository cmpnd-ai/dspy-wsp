The edited counterfactual outputs agree with the prediction. I’ll now test a separate, low-rank query-side correction, restricted to directions orthogonal to the validated key edit. This asks whether the final decision token contains useful evidence that the original query projection fails to use.

```python
# Hypothesis: an orthogonal low-rank query correction can read residual decision-state evidence without interfering with the validated key repair.
cf_edit=load_unlabelled_read(12)
cf_error=center_options(cf_edit["L"]-(cf["L"]+cfC),cf["mask"])[cf["mask"]].abs().max().item()
print("Actual versus predicted counterfactual relative-logit max error:",cf_error)
lognote(f"Pass 12 verifies counterfactual head effects, max centered-logit prediction error {cf_error:.6g}. "
        "The validated edit amplifies relevant cues: delivery exceptions 0.077->1.448, "
        "due date 1.155->4.758, constraint violations .728->3.157 mean signed log-odds. "
        "Allowlist .114->.378, but 1/10 edited responses reverses, a known limitation.")
# Use fixed PCA subspaces from fitting inputs only. No labels enter subspace selection.
rd,rk=64,16
DP=dvec[:,-rd:].float()/deig[-rd:].clamp_min(1e-8).sqrt().float()[None,:]
B=kvec[:,-rk:].float()
B=B-u[:,None]*(u@B)[None,:]
B,_=torch.linalg.qr(B,mode="reduced")
ksd=(Kg@B).square().mean(0).sqrt().clamp_min(1e-5)
Bscaled=B/ksd[None,:]
def query_features(data):
    z=(data["d"]-dmean)@DP
    z=torch.cat([z,torch.ones(len(z),1)],1)
    key=center_options(head_k(data["o"]),data["mask"])@Bscaled
    return (key[:,:,:,None]*z[:,None,None,:]/data["temp"][:,None,None,None]).flatten(2),z
Ffit,Zfit=query_features(fit)
Fdev,Zdev=query_features(dev)
FX=Ffit[fit["mask"]]*weights[:,None]
resid=y3-Cfit/fit["temp"][:,None]
resid=center_options(resid,fit["mask"])
qy=resid[fit["mask"]]*weights
qgram=FX.T.double()@FX.double()
qrhs=FX.T.double()@qy.double()
qeig,qvec=torch.linalg.eigh(qgram)
qproj=qvec.T@qrhs
print("Query feature count",FX.shape[1],"Gram eigenvalues min/max",qeig[0].item(),qeig[-1].item())
q_candidates={}
q_results=[]
for lam in [1.,10.,100.,1000.]:
    coeff=(qvec@(qproj/(qeig+lam))).float()
    q_candidates[lam]=coeff
    for s in [.5,1.0,1.5]:
        LF=fit["L"]+Cfit+s*(Ffit@coeff)*fit["temp"][:,None]
        LD=dev["L"]+Cdev+s*(Fdev@coeff)*dev["temp"][:,None]
        sf,sd=accuracy_stats(fit,LF),accuracy_stats(dev,LD)
        q_results.append((lam,s,sf,sd))
        print(f"q ridge={lam:6g} strength={s:.1f} | fit={sf[0]:.3f} dev={sd[0]:.3f} "
              f"gain vs original={sd[1]:+.3f} SE={sd[2]:.3f} Brier={sd[3]:.3f}")
bestq=max(q_results,key=lambda r:(r[3][0],-r[3][3]))
print("Best query addition:",bestq)
lognote("Query candidate: closed-form residual ridge in 64 decision PCs (plus intercept) x 16 option-key PCs, "
        "key output basis projected orthogonal to mean-query u. Subspaces from fit activations only. "
        "Targets residual log-gold contrasts after validated rank1, no gradients/iterative fitting.\n"+
        "\n".join(f"- q lambda={lam}, strength={s}, fit={sf[0]:.4f}, dev={sd[0]:.4f}, "
                  f"gain={sd[1]:+.4f} SE={sd[2]:.4f}, Brier={sd[3]:.4f}"
                  for lam,s,sf,sd in q_results))
```
