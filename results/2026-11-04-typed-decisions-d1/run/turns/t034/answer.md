The gains differ across workflow-heavy halves, so I won’t infer generalization from the combined practice score. I’ll log the in-sample result and validate the refit against the original model.

```python
# Hypothesis: the all-practice refit retains the independently observed gain on unseen records.
paired=[]
for k in [13,14]:
    rr={r["id"]:r for r in [json.loads(s) for s in (work/f"gen_{k}_original.jsonl").read_text().splitlines()]}
    ee=[json.loads(s) for s in (work/f"gen_{k}_edited.jsonl").read_text().splitlines()]
    paired += [r["score"]-rr[r["id"]]["score"] for r in ee]
print("All-practice paired gain",np.mean(paired),np.std(paired,ddof=1)/30)
with notes.open("a") as f:
    f.write(f"\nAll900 refit passes13/14 in-sample practice gain {np.mean(paired):.5f}±{np.std(paired,ddof=1)/30:.5f}. Current edit all900 ridge2 readout. Next validation both on 300.\n")
```
