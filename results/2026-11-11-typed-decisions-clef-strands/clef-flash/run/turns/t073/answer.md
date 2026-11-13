The full repair’s features and logits can now be checked against the local calculation. I’ll then run the gated extension at half strength as a causal dose test. This changes only the 22 targeted output weights, not the shared correction or any upstream computation.


```python
# Hypothesis: the 22-neuron extension produces a graded, predictable change in cue-sensitive margins while all upstream features remain fixed.
mm18,ff18,mk18,ll18=read_probe_arrays(18)
assert mm18==target_mm and torch.equal(mk18,target_mk)
full_feature_error=float((ff18-target_ff).abs().max())
full_logit_error=float((ll18-target_l2)[target_mk].abs().max())
print("Full repair max feature difference:",full_feature_error)
print("Full repair max logit reconstruction error:",full_logit_error)
extension_neurons=repair_b2["neurons"][2:]
half_extension=[a.clone() for a in repair_b2["tensors"]]
half_extension[2].flatten()[extension_neurons]*=.5
half_extension=tuple(a.bfloat16().float() for a in half_extension)
half_probe_logits=simulate_probe_scoring(target_ff,target_mk,target_l0,half_extension)
print("Half-extension deviation from midpoint:",
      float((half_probe_logits-.5*(target_l1+target_l2))[target_mk].abs().max()))
print("Half-extension tuning diagnostic:",repair_measure(repair_simulate(half_extension),repair_tune))
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],half_extension):
    save_edit(name,arr.reshape(SHAPES[name]))
with open(WORK/"notes.md","a") as f:
    f.write(f"\n## Pass 18 verification\nBest_2 target probes: upstream max feature difference {full_feature_error:g}; local-vs-tool max logit error {full_logit_error:g}. Constraint violation counts0/1/2/4 yield critical−routine urgency margins original[-2.537,-2.402,-2.014,-0.714], shared[-2.109,-1.095,-1.003,-0.074], best_2[-2.835,+0.157,+0.766,+1.584]. Irreversible actions alone remain weak/nonmonotonic; customer needs_human responses mixed. These limitations are recorded, not hidden.\n")
    f.write("\n## Planned pass 19 — half extension causal dose\nExact edit: best_2 incoming rows/biases unchanged, shared output entries unchanged, only22 targeted outgoing weights halved (±4.1875 instead of±8.375). READ edited on156 target-cue probes, target_cue_probes_edited.json. Mechanism test only; no validation selection. best_2 saved.\n")
```
