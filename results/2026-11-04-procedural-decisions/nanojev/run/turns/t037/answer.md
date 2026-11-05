The high-only edit reaches 62.78% validation, below the full edit’s 68.67%, so I’ll discard the ablation and restore the full correction. Next I’ll report paired performance separately by workflow, to make clear where the repair helps and where limitations remain.

```python
# Hypothesis: the confirmed full correction improves all workflows, with the largest gain in sensor conditioning.
save_edit("scalar.weight",torch.from_numpy((W0+delta900).astype(np.float32)[None]))
wfs=sorted(set(x["inputs"]["workflow"] for x in data))
for j,wf in enumerate(wfs):
    ids=[x["id"] for x in data if x["inputs"]["workflow"]==wf]
    json.dump({"type":"decide","model":"both","problems":ids},open(work/f"workflow_{j}.json","w"))
with open(notes,"a") as f:
    f.write("\nPass 15 high-only validation .62778, gain .17667 ± .03058; inferior to full .68667. Rejected; restored best_900_ridge01."
            "\nPasses 16-18: full confirmed scalar correction, paired per-workflow diagnostics on all 300 practice items per workflow (in-sample): event, multi-view, sensors. No other edits.\n")
```
