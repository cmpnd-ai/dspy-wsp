The earlier rank-one repair still has the best confirmed validation score: **73.40%**, versus 72.73% for the linear-target alternative. I’ll keep it and stop fitting candidates. The remaining runs will verify its dose response, cue sensitivity, and final saved weights.

```python
# Hypothesis: the chosen repair has an optimal strength for reading state evidence, rather than merely changing confidence globally.
best2_saved = torch.load(WORK/"best_2/weights_and_direction.pt", weights_only=False)
print("Best2 stored direction agreement:", float((best2_saved["v"]-v_ref).abs().max()))
best2_matrix = torch.from_numpy(np.load(WORK/"best_2/edited/k.weight.npy")).float()
assert torch.allclose(best2_matrix, K0+torch.outer(u,v_ref), atol=3e-7, rtol=1e-6)
for dose in (0., .5, 1., 1.5):
    print("Best2 dose", dose, eval_cached(dev, dev["L"]+dose*Cdev_ref))
save_edit("k.weight", K0+.5*torch.outer(u,v_ref))
with open(WORK/"final_all_practice.json", "w") as f:
    json.dump({"type":"decide", "model":"edited", "problems":[e["id"] for e in examples]}, f)
print("Counterfactual request file keys:", list(json.load(open(WORK/"counterfactual_edited.json"))))
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 19 linear-target hidden validation: original .582000 -> edited .727333, "
            "gain .145333, paired SE .014654; 168 improved, 39 worsened. "
            "Best2 remains the best confirmed candidate (.734000 validation). "
            "No more fitting or candidate selection; remaining passes are diagnostics and final verification.\n")
    f.write("\n## Before pass 20\nExact diagnostic edit: k.weight = K0 + .5*outer(u,v_ref), "
            "where v_ref is the fixed best2 direction (log-gold floor .1, ridge .0003, fitted strength .75). "
            "All other tensors original. Paired DECIDE on the same 180 development cases in paired_dev.json. "
            "Dose response uses multipliers 0/.5/1/1.5 of this fixed direction, not refitting. "
            "Final choice remains full-strength best2.\n")
```
