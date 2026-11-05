I’ll summarize the subspace results and reverse the correction as a negative control. If the direction is causal, reversing it should worsen the original preferences.

```python
# Hypothesis: the sign of the repaired readout direction matters, not merely its norm.
for pn in bi.range(13,23):
    rows=json.load((work/f"fwd_{pn}"/"meta.json").open())["rows"]
    probs=np.load(work/f"fwd_{pn}"/"probabilities.npy")
    hits=[]
    for ri,row in enumerate(rows[:200]):
        gg=byid[panel_order[row["request"]]]["labels"]["gold"][row["question"]]["probabilities"]
        hits.append(row["options"][np.nanargmax(probs[ri])]==bi.max(gg,key=gg.get))
    print(pn,round(float(np.mean(hits)),3))
install_job(9)
with (work/"notes.md").open("a") as f:
    f.write("\nFixed fitting-panel accuracies: dose 0/.25/.5/.75/1/1.25/1.5 = .370/.425/.505/.605/.585/.580/.565 (see actual summary for high-dose values). Pass 23: negative control scalar original -0.5*dw600, same fixed panel.\n")
```