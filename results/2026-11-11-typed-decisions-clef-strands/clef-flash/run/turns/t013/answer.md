The residual scorer supplies most option separation: its centered logit RMS is 1.117, versus 0.171 for the prior and 0.092 for cosine similarity. I’ll test a closed-form correction to its final 1,024 weights, leaving all representations, gates, and scales unchanged. Regularization and edit strength will be assessed only on the tuning split.

```python
# Hypothesis: the dominant failure is a misaligned residual output direction, recoverable from existing nonlinear cue features.
fit_rows = D["isfit"]
maskfit = D["mask"][fit_rows]
Cfit = D["C"][fit_rows]
noptsfit = maskfit.sum(-1).float()
X = (Cfit/noptsfit.sqrt()[:,None,None])[maskfit].double()
gold_smooth = D["gold"].clamp_min(0.015)
target_log = center_options(gold_smooth.log(),D["mask"])
target_delta = target_log-center_options(D["logits"],D["mask"])
y = (target_delta[fit_rows]/noptsfit.sqrt()[:,None])[maskfit].double()
G = X.T@X
rhs = X.T@y
evals,evecs = torch.linalg.eigh(G)
print("Gram spectrum:", {q:round(torch.quantile(evals.clamp_min(0),q).item(),6) for q in [0.,.25,.5,.75,.9,.99,1.]})
ridge_unit = G.diag().mean().item()
print("Ridge unit:",ridge_unit,"X shape:",tuple(X.shape),"original output norm:",wout.norm().item())
candidates = {}
trial_table = []
for ratio in [0.001,0.01,0.1,1.,10.,100.]:
    lam = ratio*ridge_unit
    delta = (evecs@((evecs.T@rhs)/(evals+lam))).float()
    candidates[ratio]=delta
    for strength in [0.5,1.,1.5]:
        w = (wout+strength*delta).to(torch.bfloat16).float()
        L = D["logits"]+D["C"]@(w-wout)
        f = summary_logits(L,D["isfit"])
        t = summary_logits(L,D["istune"])
        trial_table.append((ratio,strength,f["accuracy"],t["accuracy"],t["gain"],t["se"],delta.norm().item()))
print("ratio strength fit_acc tune_acc tune_gain paired_se delta_norm")
for row in trial_table:
    print(" ".join(f"{v:.4g}" for v in row))
note("## Closed-form readout experiment\nResidual dominates option margins: centered RMS residual=1.1171, prior=0.1711, gated cosine=0.0915. Computed within-question-centered ridge correction to final residual output weights. Targets are log gold probabilities floored at 0.015; each question equally weighted. Six fixed regularizers, three strengths evaluated locally on tune, no gradients or iterative weight fitting. Backbone and all intermediate head tensors fixed.")
```
