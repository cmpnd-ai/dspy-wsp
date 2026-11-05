The 256-coordinate result also matches the replay exactly. I’ll test 512 coordinates next to complete the coordinate-ablation sequence.

```python
# Hypothesis: half of the corrected coordinates recovers more accuracy, but not the full distributed signal.
sparse512=np.asarray(w0).copy()
sparse512[rank[:512]]+=dc[rank[:512]]
save_edit("head.scorer.3.weight",torch.tensor(sparse512).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass 19 top256 .58810, gain .06825 SE .01550. Next pass 20 top512 corrections only, same 420 testids; predicted .67778.\n")
```
