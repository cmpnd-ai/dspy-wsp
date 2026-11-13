Simple centroid gates are unreliable, so I won’t use them. I’ll test a closed-form identity read that discounts state variation within each option class. It uses only question/option identities—not gold answers—and must isolate the targeted options on unseen tuning records before any gated edit is attempted.

```python
# Hypothesis: removing within-option state variation yields reliable identity gates for narrowly targeted scorer repairs.
GF=gate_features[fit_flat_mask]
gmean=GF.mean(0)
GFC=GF-gmean
GY=F.one_hot(class_ids[fit_flat_mask],len(option_classes)).double()
gymean=GY.mean(0)
GG=GFC.T@GFC
GR=GFC.T@(GY-gymean)
gev,gU=torch.linalg.eigh(GG)
gunit=GG.diag().mean().item()
gate_models={}
print("Gate ridge, fit accuracy, tune accuracy, target min-positive/max-negative")
for ratio in [1e-6,1e-5,1e-4,1e-3,.01]:
    coef=gU@((gU.T@GR)/(gev[:,None]+ratio*gunit))
    bias=gymean-gmean@coef
    scores=gate_features@coef+bias
    fitacc=(scores.argmax(-1)==class_ids)[fit_flat_mask].float().mean().item()
    tuneacc=(scores.argmax(-1)==class_ids)[D["mask"]&D["istune"][:,None]].float().mean().item()
    mins,maxs=[],[]
    for c,(qkey,o) in enumerate(option_classes):
        if qkey in target_qkeys:
            tp=(class_ids==c)&D["mask"]&D["istune"][:,None]
            tn=(class_ids!=c)&D["mask"]&D["istune"][:,None]
            mins.append(scores[:,:,c][tp].min().item())
            maxs.append(scores[:,:,c][tn].max().item())
    gate_models[ratio]=(coef.float(),bias.float(),scores.float())
    print(ratio,round(fitacc,4),round(tuneacc,4),round(min(mins),3),round(max(maxs),3))
note("Centroid-only gates rejected: tune option-identity accuracy 86.7%, several negative separation margins. Testing direct ridge identity readouts that remove within-class state covariance; labels are schema identities only, not gold answers. No weight changes.")
```

This diagnostic changes no weights; the validation-confirmed edit remains the fallback.