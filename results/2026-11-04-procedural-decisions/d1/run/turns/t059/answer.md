Reversing the direction slightly hurts accuracy, rather than reproducing the gain. I’ll restore C, check the output-bias explanation, and verify unchanged features on the second practice partition.

```python
# Hypothesis: the small raw-logit discrepancy is solely the unchanged scalar scorer bias.
bias=float(load_original("head.scorer.3.bias").item())
predL=np.einsum("qod,d->qo",A,wC)+bias
print("Unchanged output bias",bias,"bias-corrected replay error",np.nanmax(np.abs(predL-editL)))
save_edit("head.scorer.3.weight",candidateC)
req=json.loads((work/"read_test.json").read_text())
req["model"]="edited"
(work/"verify_test.json").write_text(json.dumps(req))
with (work/"notes.md").open("a") as f:
    f.write(f"\nPass 22 reversed direction .50952, gain -.01032 SE .00425. Restored C. Unchanged scorer output bias {bias}; bias-corrected replay max error {np.nanmax(np.abs(predL-editL))}. Next pass 23 edited READ 420 testids to verify second partition features. Final candidate remains C.\n")
```
