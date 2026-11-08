The doubled extension also matched the prediction and performed worse than the validated dose. Next I’ll retain each conditional channel’s average option preference but remove its state-dependent variation. This tests whether the gain comes from reading evidence rather than shifting answer priors.
```python
# Hypothesis: the conditional repair needs state-dependent directions; retaining only average option preferences will lose its gain.
prior_only_params=tuple(v.clone() for v in best2_params)
for it,(qno,oi) in enumerate(selected_gate_keys):
    u,v=cond_units[2*it:2*it+2]
    gw=(best2_params[0][u]+best2_params[0][v])/2
    gb=(best2_params[1][u]+best2_params[1][v])/2
    dd=(best2_params[0][u]-best2_params[0][v])/(2*.025)
    bb=(best2_params[1][u]-best2_params[1][v])/(2*.025)
    fq=D["fit"]&(qid==qno)
    prior=float((D["x"][fq,oi]@dd+bb).mean())
    prior_only_params[0][u]=gw
    prior_only_params[0][v]=gw
    prior_only_params[1][u]=gb+.025*prior
    prior_only_params[1][v]=gb-.025*prior
prior_only_logits=logits_scorer_params(D["x"],prior_only_params)
print("Conditional state ablation",evaluate_logits(prior_only_logits))
print("Compared with full repair",paired_local(prior_only_logits,best2_logits))
torch.save({"params":prior_only_params,"purpose":"conditional state-dependence ablation"},W/"conditional_prior_only.pt")
save_scorer_params(prior_only_params)
note("\nPass17 actual conditional2x: dev55.000% ->69.196%, +14.196pp SE1.856pp,131 improved/39 worsened; matches local prediction, below full1x70.089%.\nNext pass18: DECIDE both224 dev IDs in dose_dev.json. Exact diagnostic edit conditional_prior_only.pt: shared residual pair33/746 unchanged at1x;54 conditional gates retained, but each state direction removed and replaced by its448-fit mean correction for that q/option in the paired biases. Expected "+str(evaluate_logits(prior_only_logits))+". Best2 fully backed up.")
```
