The shared-repair forward pass is complete. I’ll verify that its features are unchanged and its logits match the local calculation, then restore the full candidate and run the identical probes. That comparison directly tests the 22 gated neurons.

```python
# Hypothesis: the measured cue-response differences arise entirely from the edited scorer neurons, not from altered upstream states.
mm17,ff17,mk17,ll17=read_probe_arrays(17)
assert mm17==target_mm and torch.equal(mk17,target_mk)
shared_feature_error=float((ff17-target_ff).abs().max())
shared_logit_error=float((ll17-target_l1)[target_mk].abs().max())
print("Shared repair max feature difference:",shared_feature_error)
print("Shared repair max logit reconstruction error:",shared_logit_error)
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],repair_b2["tensors"]):
    save_edit(name,arr.reshape(SHAPES[name]))
with open(WORK/"notes.md","a") as f:
    f.write(f"\n## Pass 17 verification\nShared repair probe max feature difference {shared_feature_error:g}; max local-vs-tool logit error {shared_logit_error:g}. Controlled target probes show strong violation→urgency repair, weaker/mixed customer-human cues; do not claim all cues fixed.\n")
    f.write("\n## Planned pass 18\nExact edit best_2 restored: shared neurons791/1015 plus22 gated field-specific neurons, tensors from targeted_candidate.pt. READ edited on same156 target-cue requests, target_cue_probes_edited.json. No labels/score; compare against passes16(original) and17(shared only).\n")
```

