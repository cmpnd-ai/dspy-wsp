The probes support a cue-sensitive repair for agent outcomes: with one tool error, the partial-versus-success margin changes from −0.875 to +0.349, while the no-error case still favors success. They also expose a limitation: invoice-ID duplication remains poorly detected.

I’ll make one final check of the targeted extension: whether its poor simulation was caused mainly by correctable rounding offsets, rather than the proposed evidence directions.

```python
# Hypothesis: the targeted extension's loss comes from bfloat16 offset error, which can be corrected analytically without refitting labels.
rounding_trials={}
for gs in [32.,64.]:
    tensors=targeted_tensors(gs,1.)
    wf,bf,wl=[t.clone() for t in tensors]
    pre=F.linear(D["features"],wf[new_neurons],bf[new_neurons])
    errstats=[]
    for k,c in enumerate(selected_classes):
        pos=(class_ids==c)&fit_flat_mask
        r,b=expert_rows[c]
        desired=(D["features"]@r+b)
        actual=gate*(F.gelu(pre[:,:,2*k])*wl[new_neurons[2*k]]+
                     F.gelu(pre[:,:,2*k+1])*wl[new_neurons[2*k+1]])
        difference=(desired-actual)[pos]
        shift=difference.mean().item()/(gate*pair_gain)
        # Match the fit-set mean of the intended analytic response; no gold labels enter this rounding correction.
        bf[new_neurons[2*k]]+=shift/2
        bf[new_neurons[2*k+1]]-=shift/2
        errstats.append((c,difference.mean().item(),difference.std().item()))
    bf=bf.to(torch.bfloat16).float()
    corrected=(wf,bf,wl)
    rounding_trials[gs]=corrected
    L=simulate_scoring_tensors(corrected)
    print("gate",gs,"target fit mean-error/std median",
          np.median(np.array(errstats)[:,1:],axis=0).round(3),
          "fit",round(summary_logits(L,D["isfit"])["accuracy"],4),
          "tune",summary_logits(L,D["istune"]))
    print("per-target error:",[(option_classes[c][1],round(mean,3),round(std,3)) for c,mean,std in errstats])
note("## Controlled probe findings\nTool-error cue is repaired: partial−success mean margin at 0 errors original −1.981 / edited −1.795; at 1 error −0.875 / +0.349; at 2 errors +0.130 / +1.121. Constraint-violation critical−routine urgency rises more strongly after edit. Allowlist+change-window reduces credential compromise margin more after edit. Duplicate-ID sensitivity is NOT repaired (present−absent margin effect shrinks); do not claim that mechanism fixed.\nChecking whether analytically compensating bfloat16 offsets rescues the targeted extension; no new label fitting.")
```

The validated edit remains unchanged during this check.
