The 1.5× run also matched the prediction and performed worse than full strength. I’ll restore the exact validation-confirmed weights and run the full-strength development comparison, completing the actual dose-response check.

```python
# Hypothesis: the fixed, validated full-strength repair performs better than either weaker or excessive amplification.
save_edit("k.weight", best2_matrix)
assert edited() == ["k.weight"]
assert torch.equal(load("k.weight"), best2_matrix)
import hashlib
final_weight_sha256 = hashlib.sha256((WORK/"edited/k.weight.npy").read_bytes()).hexdigest()
print("Restored best2; k.weight file SHA256:", final_weight_sha256)
print("Expected full-strength development:", eval_cached(dev, dev["L"]+Cdev_ref))
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 21 actual best2 1.5-dose development: original .577778 -> edited .746667, "
            "gain .168889, paired SE .022218; 107 improved, 34 worsened. "
            "Matches cached-head prediction. Restored exact full-strength best2.\n")
    f.write(f"Frozen final k.weight SHA256: {final_weight_sha256}. Only edited tensor is k.weight.\n")
    f.write("\n## Before pass 22\nExact edit: restored best_2/edited/k.weight.npy, "
            "k.weight = K0 + outer(u,v_ref), log-gold floor .1, ridge .0003, strength .75. "
            "All other tensors original. Paired DECIDE 180 development cases in paired_dev.json. "
            "Full-strength actual dose-response confirmation; predicted .772222. "
            "Weights are frozen for all remaining passes.\n")
```
