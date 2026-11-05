I’ll restore the validated edit, record the negative-control result, and verify the exact saved candidate on a larger practice panel.

```python
# Hypothesis: the validated scalar direction is the repair to retain; diagnostic variants are discarded.
rows=json.load((work/"fwd_23"/"meta.json").open())["rows"]
probs=np.load(work/"fwd_23"/"probabilities.npy")
hits=[]
for ri,row in enumerate(rows[:200]):
    gg=byid[panel_order[row["request"]]]["labels"]["gold"][row["question"]]["probabilities"]
    hits.append(row["options"][np.nanargmax(probs[ri])]==bi.max(gg,key=gg.get))
print("Reversed direction accuracy:",float(np.mean(hits)))
save_edit("scalar.weight",torch.from_numpy(np.load(work/"best_1"/"scalar.weight.npy")).clone())
print("cosine with original head:",float(torch.nn.functional.cosine_similarity(load("scalar.weight"),load_original("scalar.weight")).item()))
json.dump({"type":"decide","model":"both","problems":fit_ids[:450]},(work/"final_practice.json").open("w"))
with (work/"notes.md").open("a") as f:
    f.write(f"\nCorrection: dose 1.25/1.5 accuracies were .570/.575. Covariance cutoff 1/.1/.01 accuracies .515/.560/.580. Reversed direction accuracy {np.mean(hits):.3f}. Restored best_1 exactly; no diagnostic variant retained.\nPass 24: paired decide, restored scalar original+dw600_linear, first 450 fitting cases. Descriptive fit performance only; independent test and validation remain the evidence for selection.\n")
```