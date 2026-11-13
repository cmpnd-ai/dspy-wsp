The input features contain more usable information: a direct linear correction improves tuning accuracy by up to 3.6 points, while reweighting existing neurons does not. I’ll check whether that gain is broad or concentrated, and whether emphasizing the gold winning option helps the accuracy metric without losing generalization.

```python
# Hypothesis: scorer compression loses task-specific ranking information, especially where gold distributions have narrow winning margins.
fdelta = fcandidates[1e-6]
Lraw = D["logits"]+0.5*(D["FC"]@fdelta)
print("Raw-input candidate, tune by workflow:")
print(summary_logits(Lraw,D["istune"],True))
for (wf,q) in sorted(groups):
    ix = torch.tensor([byid[i]["inputs"]["workflow"]==wf and m["question"]==q for i,m in zip(D["ids"],D["meta"])]) & D["istune"]
    b = summary_logits(D["logits"],ix)
    t = summary_logits(Lraw,ix)
    print(wf[:8],q,round(b["accuracy"],3),"->",round(t["accuracy"],3))

hard = F.one_hot(D["target"],5).float()*D["mask"]
mixed_gold = 0.5*D["gold"]+0.5*hard
alt_targets = {
    "mixed":center_options(mixed_gold.clamp_min(.015).log(),D["mask"]),
    "hard_margin":center_options(hard*2.5,D["mask"])
}
alt_candidates = {}
print("\nAlternative closed-form targets: target ridge strength fit tune gain se")
for target_name,T in alt_targets.items():
    Y = ((T-center_options(D["logits"],D["mask"]))[fit_rows]/noptsfit.sqrt()[:,None])[maskfit].double()
    R = FX.T@Y
    for ratio in [1e-5,1e-4,1e-3]:
        d = (fevecs@((fevecs.T@R)/(fevals+ratio*funit))).float()
        alt_candidates[target_name,ratio]=d
        for strength in [.5,1.]:
            L=D["logits"]+strength*(D["FC"]@d)
            f,t=summary_logits(L,D["isfit"]),summary_logits(L,D["istune"])
            print(target_name,ratio,strength,*[round(z,4) for z in [f["accuracy"],t["accuracy"],t["gain"],t["se"]]])
field = D["features"][:,0,:1024]
opt = D["features"][:,:,1024:2048]
print("Field coordinate RMS and across-question std:",field.square().mean().sqrt().item(),field.std(0).square().mean().sqrt().item())
print("Option coordinate RMS and across-option std:",opt[D["mask"]].square().mean().sqrt().item(),opt[D["mask"]].std(0).square().mean().sqrt().item())
note("Raw-feature correction outperforms existing-activation correction: best initial tune accuracy 0.746 (+0.036, paired SE 0.01514), with fit 0.7796, strength 0.5. Testing two fixed alternative targets because scoring uses argmax, not gold calibration. No edit saved yet.")
```
