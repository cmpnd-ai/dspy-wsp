Changing only the scorer’s output weights did not help tuning accuracy, so I won’t keep those corrections. The activations are nearly low-rank, suggesting the first scorer layer may discard distinctions that remain in its input. I’ll test that by measuring how well the richer input features explain the same option-margin errors.


```python
# Hypothesis: useful option distinctions exist before residual_scorer.0 but are compressed away by its nearly low-rank activation map.
D["features"] = torch.cat([a["head_inputs"] for a in batches],0)
D["FC"] = center_options(D["features"][:,:,1024:],D["mask"])
FX = (D["FC"][fit_rows]/noptsfit.sqrt()[:,None,None])[maskfit].double()
FG = FX.T@FX
Frhs = FX.T@y
fevals,fevecs = torch.linalg.eigh(FG)
funit = FG.diag().mean().item()
print("Raw feature Gram spectrum:", {q:round(torch.quantile(fevals.clamp_min(0),q).item(),6) for q in [0.,.25,.5,.75,.9,.99,1.]})
print("Raw feature ridge unit:",funit)
fcandidates, ftable = {}, []
for ratio in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]:
    delta = (fevecs@((fevecs.T@Frhs)/(fevals+ratio*funit))).float()
    fcandidates[ratio] = delta
    for strength in [0.5,1.,1.5]:
        L = D["logits"]+strength*(D["FC"]@delta)
        f,t = summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
        ftable.append((ratio,strength,f["accuracy"],t["accuracy"],t["gain"],t["se"],delta.norm().item()))
print("ratio strength fit_acc tune_acc tune_gain paired_se correction_norm")
for row in ftable:
    print(" ".join(f"{v:.5g}" for v in row))
contrib = center_options(D["activations"],D["mask"])*wout[None,None,:]*gate
nrms = contrib[D["mask"]].square().mean(0).sqrt()
least = torch.argsort(nrms)
print("Least influential existing neurons:",[(i,round(nrms[i].item(),8),round(wout[i].item(),8)) for i in least[:12].tolist()])
print("Strongest output neuron contributions:",[(i,round(nrms[i].item(),4)) for i in torch.argsort(nrms,descending=True)[:10].tolist()])
note("## Final-output-only correction rejected\nAll tested direct corrections to residual_scorer.3.weight failed tune (best -0.2 percentage points; strongest -4.2 pp). No edit saved. Activation Gram spectrum is severely compressed (median eigenvalue 0.000868 versus maximum 92446.9). Testing linear corrections directly from the existing field/option interaction input to localize this bottleneck.")
```
