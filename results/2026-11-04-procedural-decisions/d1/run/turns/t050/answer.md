The 64-coordinate ablation matched the replay prediction and reduced accuracy. I’ll now retain 256 coordinates; the predicted score is 58.81%, still well below full C’s 80.63% on these cases.

```python
# Hypothesis: restoring more of the distributed readout correction yields a partial recovery.
sparse256=np.asarray(w0).copy()
sparse256[rank[:256]]+=dc[rank[:256]]
save_edit("head.scorer.3.weight",torch.tensor(sparse256).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass 18 top64 .47619, gain -.04365 SE .01041. Next pass 19 top256 corrections only, same 420 testids; predicted .58810.\n")
```
