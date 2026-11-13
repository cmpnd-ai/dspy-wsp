Half strength produced the predicted gain. I’ll now test 1.5 times the chosen repair on the same development cases to check whether strengthening this readout too far reduces accuracy.

```python
# Hypothesis: over-amplifying the repaired evidence direction will reduce accuracy relative to its validated strength.
save_edit("k.weight", K0 + 1.5 * torch.outer(u, v_ref))
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 20 actual best2 half-dose development: original .577778 -> edited .697778, "
            "gain .120000, paired SE .014321; 86 improved, 15 worsened. "
            "Matches cached-head prediction exactly.\n")
    f.write("\n## Before pass 21\nExact diagnostic edit: k.weight = K0 + 1.5*outer(u,v_ref), "
            "using the fixed best2 direction; all other tensors original. "
            "Paired DECIDE same 180 development cases in paired_dev.json. "
            "Cached prediction .746667, below full-strength .772222. "
            "No refitting or candidate selection; full-strength best2 remains final choice.\n")
print("Predicted 1.5-dose development:", eval_cached(dev, dev["L"] + 1.5*Cdev_ref))
```
