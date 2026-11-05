I’ll compute the aligned dose-response, then test a 1.25× correction to see whether gains saturate.

```python
# Hypothesis: the repair has a bounded useful strength rather than acting as generic sharpening.
for pn in [13,14,15,16,17]:
    pm=json.load((work/f"fwd_{pn}"/"meta.json").open())["rows"]
    probs=np.load(work/f"fwd_{pn}"/"probabilities.npy")
    hits=[]
    for ri,row in enumerate(pm[:200]):
        gg=byid[panel_order[row["request"]]]["labels"]["gold"][row["question"]]["probabilities"]
        hits.append(row["options"][np.nanargmax(probs[ri])]==bi.max(gg,key=gg.get))
    print(pn, "panel accuracy", np.mean(hits))
for ri,row in enumerate(pm[200:],200):
    if row["question"] in ["action","category","risk"]:
        print(synlabels[row["request"]-40],row["question"],dict(zip(row["options"],np.round(probs[ri],3))))
install_job(4)
with (work/"notes.md").open("a") as f:
    f.write("\nPass 18 (corrected numbering): scalar original + 1.25*dw600; fixed 40-case fitting panel and 7 synthetic requests.\n")
```