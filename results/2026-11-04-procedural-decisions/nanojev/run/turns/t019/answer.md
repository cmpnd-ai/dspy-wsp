As predicted, 1.5× overshoots: accuracy falls from 66.89% to 61.70%. I’ll restore the validated full-strength edit and extract the remaining practice features in two batches. Those features will let me separate question/answer priors from within-question factual evidence and assess whether more fitting data helps.

```python
# Hypothesis: additional practice features can stabilize the evidence readout, especially intent and urgency.
save_edit("scalar.weight",torch.from_numpy((W0+delta).astype(np.float32)[None]))
remaining=[r["id"] for r in records if r["id"] not in set(fit_ids)]
read_groups=[remaining[:240],remaining[240:]]
for j,ids in enumerate(read_groups):
    json.dump({"type":"read","model":"original","from":{"pass":1,"ids":ids}},open(work/f"read_remaining_{j}.json","w"))
with open(notes,"a") as f:
    f.write("\nPass 6 1.5x overshoot: .61704 vs .47185, gain .14519 ± .02819, worse than 1x (.66889). Restored validated 1x."
            "\nPasses 7-8: original-model read on remaining 480 practice records in two 240-record batches; all existing features retained. No new edit.\n")
```
