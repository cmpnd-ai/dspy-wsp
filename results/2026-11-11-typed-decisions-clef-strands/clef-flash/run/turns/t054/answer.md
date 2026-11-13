The block ablations show that the repair depends on a balanced combination of option and field–option interactions; changing any one block hurts. I’ll leave that direction intact and examine the original scorer’s nearly inactive neurons. If they contain useful input directions but are suppressed by their activation offsets, those offsets may be another repairable bottleneck.


```python
# Hypothesis: dormant scorer neurons discard informative feature directions because their preactivations sit far below GELU's active range.
repair_pre0=F.linear(repair_features,repair_W0,repair_b0)
repair_act0=F.gelu(repair_pre0)
repair_con0=center_options(repair_act0,repair_mask)*repair_wout[None,None,:]*repair_gate
repair_rms0=repair_con0[repair_fit][repair_mask[repair_fit]].square().mean(0).sqrt()
repair_mu0=repair_pre0[repair_fit][repair_mask[repair_fit]].mean(0)
repair_std0=repair_pre0[repair_fit][repair_mask[repair_fit]].std(0)
repair_max0=repair_pre0[repair_fit][repair_mask[repair_fit]].max(0).values
repair_min0=repair_pre0[repair_fit][repair_mask[repair_fit]].min(0).values
repair_used=((repair_current_tensors[0]!=repair_W0).any(1)|(repair_current_tensors[1]!=repair_b0))
repair_dormant=(repair_rms0<1e-5)&~repair_used
print("Unused dormant neurons:",int(repair_dormant.sum()))
for label,vals in [("preactivation mean",repair_mu0),("preactivation std",repair_std0),
                   ("preactivation max",repair_max0),("incoming norm",repair_W0.norm(dim=1)),
                   ("outgoing magnitude",repair_wout.abs())]:
    print(label, "all percentiles",torch.quantile(vals,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
          "dormant",torch.quantile(vals[repair_dormant],torch.tensor([0.,.1,.5,.9,1.])).tolist())
repair_sorted_dormant=torch.where(repair_dormant)[0][torch.argsort(repair_rms0[repair_dormant])]
print("Sample dormant neurons (id, mean, std, max, option contribution rms):",
      [(int(i),round(float(repair_mu0[i]),3),round(float(repair_std0[i]),3),round(float(repair_max0[i]),3),float(repair_rms0[i])) for i in repair_sorted_dormant[:12]])
print("Best2 still current:",edited())
with open(WORK/"notes.md","a") as f:
    f.write("\n## Dormant-neuron mechanism test planned\nAll independent option/interaction block reweightings hurt tuning, so best_2 direction retained. Inspecting unused original scorer neurons with <1e-5 option-centered contribution RMS. Hypothesis: large negative preactivation offsets suppress otherwise useful original directions. Will test centered dormant activations with a one-shot output ridge correction, not iterative fitting.\n")
```
